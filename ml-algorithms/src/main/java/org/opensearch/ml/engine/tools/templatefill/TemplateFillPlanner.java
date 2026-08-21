/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.engine.tools.templatefill;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.opensearch.core.action.ActionListener;
import org.opensearch.ml.common.FunctionName;
import org.opensearch.ml.common.agenticsearch.AgenticSearchTemplate;
import org.opensearch.ml.common.dataset.remote.RemoteInferenceInputDataSet;
import org.opensearch.ml.common.input.MLInput;
import org.opensearch.ml.common.output.MLOutput;
import org.opensearch.ml.common.output.model.ModelTensorOutput;
import org.opensearch.ml.common.output.model.ModelTensors;
import org.opensearch.ml.common.transport.prediction.MLPredictionTaskAction;
import org.opensearch.ml.common.transport.prediction.MLPredictionTaskRequest;
import org.opensearch.ml.common.utils.StringUtils;
import org.opensearch.ml.engine.processor.MLProcessorType;
import org.opensearch.ml.engine.processor.ProcessorChain;
import org.opensearch.transport.client.Client;

import com.jayway.jsonpath.JsonPath;

import lombok.extern.log4j.Log4j2;

/**
 * Fills one registered search template for a question and returns the rendered search body.
 *
 * <p>The path is deliberately short: resolve the schema, force one tool call, validate what
 * comes back, render. No index mapping and no sample document are fetched, because the
 * params were fixed at registration — skipping those reads is half of why this path is
 * faster than generating DSL.
 *
 * <p>Every failure ends the same way: {@code onFailure}, for the caller to degrade to
 * free-DSL generation. That includes the model deciding the template cannot express the
 * question, which is a healthy outcome rather than an error — a near-miss fill would
 * return a query that looks fine and answers the wrong question, which is the exact failure
 * the abstain flag exists to prevent. Abstention is logged at info; everything else at warn.
 *
 * <p>Structural validity is what this guarantees, not semantic correctness: the body's shape
 * comes from a template that was render-checked at registration, so the query parses, but
 * whether it means what the user asked is still the model's judgment.
 */
@Log4j2
public class TemplateFillPlanner {

    public static final String QUESTION_FIELD = "question";
    public static final String SYSTEM_PROMPT_FIELD = "system_prompt";
    public static final String USER_PROMPT_FIELD = "user_prompt";
    public static final String FILL_SCHEMA_FIELD = "fill_schema";
    public static final String LLM_INTERFACE_FIELD = "_llm_interface";
    /** Plainer alias, since an operator sets this on the tool spec rather than the agent's LLM. */
    public static final String LLM_INTERFACE_ALIAS_FIELD = "llm_interface";
    public static final String TENANT_ID_FIELD = "tenant_id";

    /** Params the model must never see: agent scaffolding, and our own injected plumbing. */
    private static final List<String> NON_MODEL_PARAMS = List.of("index_mapping", "sample_document", "template", "search_templates");

    private final Client client;
    private final TemplateSchemaResolver schemaResolver;
    private final StoredTemplateRenderer renderer;
    /** Digs a JSON object out of a text response on the prompt-enforced path. */
    private final ProcessorChain extractJson;

    public TemplateFillPlanner(Client client, TemplateSchemaResolver schemaResolver, StoredTemplateRenderer renderer) {
        this.client = client;
        this.schemaResolver = schemaResolver;
        this.renderer = renderer;
        List<Map<String, Object>> configs = new ArrayList<>();
        configs.add(Map.of("type", MLProcessorType.EXTRACT_JSON.getValue(), "extract_type", "object"));
        this.extractJson = new ProcessorChain(configs);
    }

    /**
     * Fill {@code templateId} for the question in {@code parameters}.
     *
     * @param templateId the registered template, which is also the stored script id
     * @param modelId the LLM to call
     * @param parameters the tool's execute parameters; supplies the question and any
     *     connector placeholders, and is not mutated
     * @param listener yields the rendered search body, or fails so the caller can fall back
     */
    public void fill(String templateId, String modelId, Map<String, String> parameters, ActionListener<Map<String, Object>> listener) {
        try {
            String question = parameters.get(QUESTION_FIELD);
            if (question == null || question.isBlank()) {
                listener.onFailure(new IllegalArgumentException("question is required to fill a template"));
                return;
            }
            schemaResolver.resolve(templateId, ActionListener.wrap(template -> {
                try {
                    callModel(template, modelId, parameters, listener);
                } catch (Exception e) {
                    degrade(templateId, e, listener);
                }
            }, e -> degrade(templateId, e, listener)));
        } catch (Exception e) {
            degrade(templateId, e, listener);
        }
    }

    private void callModel(
        AgenticSearchTemplate template,
        String modelId,
        Map<String, String> parameters,
        ActionListener<Map<String, Object>> listener
    ) {
        Map<String, Object> paramSchema = template.getParamSchema();
        Map<String, Object> inputSchema = FillToolSchemaBuilder.buildInputSchema(paramSchema);
        String llmInterface = llmInterface(parameters);
        boolean forced = ForcedToolCall.shouldAttemptForcedTool(llmInterface);

        Map<String, String> predictionParams = new HashMap<>(parameters);
        NON_MODEL_PARAMS.forEach(predictionParams::remove);
        if (forced) {
            predictionParams.put(SYSTEM_PROMPT_FIELD, TemplateFillPromptTemplate.FILL_SYSTEM_PROMPT);
            predictionParams.put(USER_PROMPT_FIELD, TemplateFillPromptTemplate.FILL_USER_PROMPT);
            predictionParams
                .put(
                    ForcedToolCall.TOOL_CONFIG_PARAM,
                    ForcedToolCall
                        .toolConfigJson(FillToolSchemaBuilder.FILL_TOOL_NAME, FillToolSchemaBuilder.FILL_TOOL_DESCRIPTION, inputSchema)
                );
        } else {
            // This interface is known not to force tools, so the contract the tool schema
            // would have imposed has to be carried by the prompt instead. Still fills
            // templates; just without the schema-level guarantee.
            log.debug("llm interface '{}' cannot force a tool call; filling via prompt-enforced JSON", llmInterface);
            predictionParams.put(SYSTEM_PROMPT_FIELD, TemplateFillPromptTemplate.FILL_SYSTEM_PROMPT_JSON_FALLBACK);
            predictionParams.put(USER_PROMPT_FIELD, TemplateFillPromptTemplate.FILL_USER_PROMPT_JSON_FALLBACK);
            predictionParams.put(FILL_SCHEMA_FIELD, StringUtils.gson.toJson(inputSchema));
        }

        MLInput mlInput = MLInput
            .builder()
            .algorithm(FunctionName.REMOTE)
            .inputDataset(RemoteInferenceInputDataSet.builder().parameters(predictionParams).build())
            .build();
        MLPredictionTaskRequest request = MLPredictionTaskRequest
            .builder()
            .modelId(modelId)
            .mlInput(mlInput)
            .tenantId(parameters.get(TENANT_ID_FIELD))
            .build();

        String templateId = template.getTemplateId();
        client.execute(MLPredictionTaskAction.INSTANCE, request, ActionListener.wrap(response -> {
            try {
                Map<String, Object> emitted = readFill(response.getOutput(), ForcedToolCall.resultPath(llmInterface));
                complete(template, paramSchema, emitted, listener);
            } catch (Exception e) {
                degrade(templateId, e, listener);
            }
        }, e -> degrade(templateId, e, listener)));
    }

    private void complete(
        AgenticSearchTemplate template,
        Map<String, Object> paramSchema,
        Map<String, Object> emitted,
        ActionListener<Map<String, Object>> listener
    ) {
        String templateId = template.getTemplateId();
        // Read the abstain flag before anything else, and strip it either way: it is not a
        // Mustache param and must never reach the renderer. When the template owns a real
        // param of that name the hatch is disabled, so the value is left alone.
        if (!FillToolSchemaBuilder.abstainDisabled(paramSchema)) {
            Object abstain = emitted.remove(FillToolSchemaBuilder.CANNOT_EXPRESS_FIELD);
            if (isTrue(abstain)) {
                log.info("Template '{}' cannot express the question; falling back to query generation", templateId);
                listener.onFailure(new TemplateCannotExpressException(templateId));
                return;
            }
        }
        Map<String, Object> params;
        Map<String, Object> rendered;
        try {
            params = FillValidator.validate(paramSchema, emitted);
            rendered = renderer.render(templateId, params);
        } catch (Exception e) {
            degrade(templateId, e, listener);
            return;
        }
        log.debug("Template '{}' filled {} params", templateId, params.size());
        listener.onResponse(rendered);
    }

    /**
     * Pull the fill out of the model response. On the forced path the arguments sit at a
     * known path in the raw response, already structured. Otherwise the response is text
     * that may carry prose around the JSON, so it goes through the same extract-json
     * processor the rest of the codebase uses, then gets unwrapped from its
     * {@code params} envelope.
     */
    @SuppressWarnings("unchecked")
    Map<String, Object> readFill(MLOutput output, String resultPath) {
        Map<String, ?> dataAsMap = firstTensorData(output);
        if (resultPath != null) {
            Object raw = null;
            try {
                raw = JsonPath.read(dataAsMap, resultPath);
            } catch (Exception e) {
                // There is no tool-use block at all. Either the connector's PREDICT action
                // does not set supports_structured_output, so the toolConfig was never
                // injected, or the model answered in text anyway. Read it as text below
                // rather than spending another round trip.
                log.debug("no tool call at {}; reading the response as text", resultPath);
            }
            if (raw != null) {
                // A tool-use block that came back empty is a different thing from no tool-use
                // block, and worth saying so rather than reinterpreting the envelope as text.
                Map<String, Object> arguments = asMap(raw);
                if (arguments == null || arguments.isEmpty()) {
                    throw new IllegalArgumentException("forced tool call produced no input");
                }
                return arguments;
            }
        }
        Object text = dataAsMap.containsKey("response") ? dataAsMap.get("response") : dataAsMap;
        Map<String, Object> envelope = asMap(extractJson.process(StringUtils.toJson(text)));
        if (envelope == null || envelope.isEmpty()) {
            throw new IllegalArgumentException("model returned no fill");
        }
        Object params = envelope.get("params");
        Map<String, Object> fill = params instanceof Map ? new HashMap<>((Map<String, Object>) params) : new HashMap<>(envelope);
        // The abstain flag rides beside params in the envelope, so lift it in next to them.
        if (envelope.containsKey(FillToolSchemaBuilder.CANNOT_EXPRESS_FIELD)) {
            fill.put(FillToolSchemaBuilder.CANNOT_EXPRESS_FIELD, envelope.get(FillToolSchemaBuilder.CANNOT_EXPRESS_FIELD));
        }
        return fill;
    }

    /**
     * The model's interface, under either the underscore-prefixed key the agent runners use or
     * the plainer alias, since for a FLOW agent this only arrives if an operator set it on the
     * tool spec.
     */
    private static String llmInterface(Map<String, String> parameters) {
        String llmInterface = parameters.get(LLM_INTERFACE_FIELD);
        if (llmInterface == null || llmInterface.isBlank()) {
            llmInterface = parameters.get(LLM_INTERFACE_ALIAS_FIELD);
        }
        return llmInterface;
    }

    /** A JSON object as a mutable map, whether it arrives already parsed or as a string. */
    @SuppressWarnings("unchecked")
    private static Map<String, Object> asMap(Object value) {
        if (value == null) {
            return null;
        }
        if (value instanceof Map) {
            return new HashMap<>((Map<String, Object>) value);
        }
        Map<String, Object> parsed = StringUtils.fromJson(StringUtils.toJson(value), "response");
        return parsed == null ? null : new HashMap<>(parsed);
    }

    private static Map<String, ?> firstTensorData(MLOutput output) {
        if (!(output instanceof ModelTensorOutput)) {
            throw new IllegalArgumentException("unexpected model output: " + (output == null ? "null" : output.getClass().getSimpleName()));
        }
        List<ModelTensors> outputs = ((ModelTensorOutput) output).getMlModelOutputs();
        if (outputs == null
            || outputs.isEmpty()
            || outputs.get(0).getMlModelTensors() == null
            || outputs.get(0).getMlModelTensors().isEmpty()) {
            throw new IllegalArgumentException("model returned no tensors");
        }
        Map<String, ?> dataAsMap = outputs.get(0).getMlModelTensors().get(0).getDataAsMap();
        if (dataAsMap == null || dataAsMap.isEmpty()) {
            throw new IllegalArgumentException("model returned an empty response");
        }
        return dataAsMap;
    }

    /**
     * The abstain flag arrives as unvalidated JSON on the prompt-enforced path, so plain
     * truthiness will not do: the string {@code "false"} is truthy and would abstain on
     * every request.
     */
    static boolean isTrue(Object value) {
        if (value instanceof Boolean) {
            return (Boolean) value;
        }
        if (value instanceof Number) {
            return ((Number) value).doubleValue() == 1d;
        }
        if (value instanceof String) {
            String s = ((String) value).trim();
            return "true".equalsIgnoreCase(s) || "yes".equalsIgnoreCase(s) || "1".equals(s);
        }
        return false;
    }

    private void degrade(String templateId, Exception e, ActionListener<Map<String, Object>> listener) {
        log.warn("Template fill for '{}' failed ({}); falling back to query generation", templateId, e.getMessage());
        listener.onFailure(e);
    }

    /** Signals that the model judged the template unable to express the question. */
    public static class TemplateCannotExpressException extends IllegalArgumentException {
        public TemplateCannotExpressException(String templateId) {
            super("template '" + templateId + "' cannot express the question");
        }
    }
}
