/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.engine.tools.templatefill;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import org.junit.Before;
import org.junit.Test;
import org.mockito.ArgumentCaptor;
import org.mockito.Mock;
import org.mockito.MockitoAnnotations;
import org.mockito.stubbing.Answer;
import org.opensearch.core.action.ActionListener;
import org.opensearch.ml.common.agenticsearch.AgenticSearchTemplate;
import org.opensearch.ml.common.output.model.ModelTensor;
import org.opensearch.ml.common.output.model.ModelTensorOutput;
import org.opensearch.ml.common.output.model.ModelTensors;
import org.opensearch.ml.common.transport.MLTaskResponse;
import org.opensearch.ml.common.transport.prediction.MLPredictionTaskAction;
import org.opensearch.ml.common.transport.prediction.MLPredictionTaskRequest;
import org.opensearch.ml.common.utils.StringUtils;
import org.opensearch.transport.client.Client;

public class TemplateFillPlannerTests {

    private static final String BEDROCK = "bedrock/converse/claude";
    /** An interface known not to support forcing a tool. */
    private static final String UNFORCEABLE = "openai/v1/chat/completions";
    private static final String TEMPLATE_ID = "product_search";

    @Mock
    private Client client;
    @Mock
    private TemplateSchemaResolver schemaResolver;
    @Mock
    private StoredTemplateRenderer renderer;

    private TemplateFillPlanner planner;

    @Before
    public void setup() {
        MockitoAnnotations.openMocks(this);
        planner = new TemplateFillPlanner(client, schemaResolver, renderer);
    }

    // ---- happy path --------------------------------------------------------

    @Test
    public void fill_forcesTheToolAndRendersTheResult() {
        stubSchema(schemaWithQueryAndSize());
        stubToolUse(Map.of("lex_query", "tents", "size", 5.0d));
        when(renderer.render(any(), any())).thenReturn(Map.of("size", 5L, "query", Map.of("match", Map.of("title", "tents"))));

        Result result = fill(params("cheap tents", BEDROCK));

        assertNull(result.error);
        assertEquals(Map.of("size", 5L, "query", Map.of("match", Map.of("title", "tents"))), result.body);

        @SuppressWarnings("unchecked")
        ArgumentCaptor<Map<String, Object>> renderParams = ArgumentCaptor.forClass(Map.class);
        verify(renderer).render(org.mockito.ArgumentMatchers.eq(TEMPLATE_ID), renderParams.capture());
        // The integral rule survives end to end: 5.0 from the wire renders as 5.
        assertEquals(5L, renderParams.getValue().get("size"));
        assertEquals("tents", renderParams.getValue().get("lex_query"));
    }

    /** One tool, and a toolChoice naming it, so the model has nothing else to choose. */
    @Test
    public void fill_sendsAForcedToolConfig() {
        stubSchema(schemaWithQueryAndSize());
        stubToolUse(Map.of("lex_query", "tents"));
        when(renderer.render(any(), any())).thenReturn(Map.of("query", Map.of()));

        fill(params("cheap tents", BEDROCK));

        Map<String, String> sent = capturePredictionParams();
        String toolConfig = sent.get(ForcedToolCall.TOOL_CONFIG_PARAM);
        assertNotNull(toolConfig);
        assertTrue(toolConfig.contains("\"toolChoice\""));
        assertTrue(toolConfig.contains(FillToolSchemaBuilder.FILL_TOOL_NAME));
        assertTrue(toolConfig.contains("lex_query"));
        assertTrue(toolConfig.contains(FillToolSchemaBuilder.CANNOT_EXPRESS_FIELD));
        assertEquals(TemplateFillPromptTemplate.FILL_SYSTEM_PROMPT, sent.get(TemplateFillPlanner.SYSTEM_PROMPT_FIELD));
    }

    /**
     * The fill prompt deliberately carries no mapping or sample document; leaving those out
     * is where the output-token saving comes from.
     */
    @Test
    public void fill_doesNotSendMappingOrSampleDocument() {
        stubSchema(schemaWithQueryAndSize());
        stubToolUse(Map.of("lex_query", "tents"));
        when(renderer.render(any(), any())).thenReturn(Map.of("query", Map.of()));

        Map<String, String> parameters = params("cheap tents", BEDROCK);
        parameters.put("index_mapping", "{\"properties\":{}}");
        parameters.put("sample_document", "{\"title\":\"x\"}");
        planner.fill(TEMPLATE_ID, "model-1", parameters, ActionListener.wrap(b -> {}, e -> {}));

        Map<String, String> sent = capturePredictionParams();
        assertFalse(sent.containsKey("index_mapping"));
        assertFalse(sent.containsKey("sample_document"));
    }

    /** The result path is for local extraction only and must not travel to the connector. */
    @Test
    public void fill_doesNotLeakTheResultPathToTheConnector() {
        stubSchema(schemaWithQueryAndSize());
        stubToolUse(Map.of("lex_query", "tents"));
        when(renderer.render(any(), any())).thenReturn(Map.of("query", Map.of()));

        fill(params("cheap tents", BEDROCK));

        assertFalse(capturePredictionParams().containsKey("_structured_output_result_path"));
    }

    // ---- abstention --------------------------------------------------------

    @Test
    public void fill_whenModelAbstains_failsWithoutRendering() {
        stubSchema(schemaWithQueryAndSize());
        stubToolUse(Map.of(FillToolSchemaBuilder.CANNOT_EXPRESS_FIELD, true));
        Result result = fill(params("how many products per brand?", BEDROCK));

        assertTrue(result.error instanceof TemplateFillPlanner.TemplateCannotExpressException);
        verify(renderer, never()).render(any(), any());
    }

    /** "false" is a string and truthy in Java; reading it naively would abstain on every request. */
    @Test
    public void fill_whenAbstainIsStringFalse_stillFills() {
        stubSchema(schemaWithQueryAndSize());
        Map<String, Object> emitted = new HashMap<>();
        emitted.put("lex_query", "tents");
        emitted.put(FillToolSchemaBuilder.CANNOT_EXPRESS_FIELD, "false");
        stubToolUse(emitted);
        when(renderer.render(any(), any())).thenReturn(Map.of("query", Map.of()));

        Result result = fill(params("cheap tents", BEDROCK));

        assertNull(result.error);
        assertNotNull(result.body);
    }

    /** The abstain flag is not a Mustache param, so it must never reach the renderer. */
    @Test
    public void fill_stripsTheAbstainFlagBeforeRendering() {
        stubSchema(schemaWithQueryAndSize());
        Map<String, Object> emitted = new HashMap<>();
        emitted.put("lex_query", "tents");
        emitted.put(FillToolSchemaBuilder.CANNOT_EXPRESS_FIELD, false);
        stubToolUse(emitted);
        when(renderer.render(any(), any())).thenReturn(Map.of("query", Map.of()));

        fill(params("cheap tents", BEDROCK));

        @SuppressWarnings("unchecked")
        ArgumentCaptor<Map<String, Object>> renderParams = ArgumentCaptor.forClass(Map.class);
        verify(renderer).render(any(), renderParams.capture());
        assertFalse(renderParams.getValue().containsKey(FillToolSchemaBuilder.CANNOT_EXPRESS_FIELD));
    }

    /** When the template owns a real param of that name the hatch is off, so the value is a fill. */
    @Test
    public void fill_whenTemplateOwnsTheAbstainName_treatsItAsAParam() {
        Map<String, Object> schema = new LinkedHashMap<>();
        schema.put(FillToolSchemaBuilder.CANNOT_EXPRESS_FIELD, Map.of("type", "string"));
        stubSchema(schema);
        stubToolUse(Map.of(FillToolSchemaBuilder.CANNOT_EXPRESS_FIELD, "a real value"));
        when(renderer.render(any(), any())).thenReturn(Map.of("query", Map.of()));

        Result result = fill(params("anything", BEDROCK));

        assertNull(result.error);
        @SuppressWarnings("unchecked")
        ArgumentCaptor<Map<String, Object>> renderParams = ArgumentCaptor.forClass(Map.class);
        verify(renderer).render(any(), renderParams.capture());
        assertEquals("a real value", renderParams.getValue().get(FillToolSchemaBuilder.CANNOT_EXPRESS_FIELD));
    }

    // ---- the fallback contract ---------------------------------------------

    @Test
    public void fill_whenSchemaIsUnresolvable_fails() {
        doAnswer((Answer<Void>) inv -> {
            ActionListener<AgenticSearchTemplate> l = inv.getArgument(1);
            l.onFailure(new IllegalArgumentException("no param-schema registered for template_id 'product_search'"));
            return null;
        }).when(schemaResolver).resolve(any(), any());

        Result result = fill(params("cheap tents", BEDROCK));

        assertNotNull(result.error);
        verify(client, never()).execute(any(), any(), any());
    }

    @Test
    public void fill_whenQuestionIsMissing_failsWithoutReadingTheSchema() {
        Map<String, String> parameters = new HashMap<>();
        parameters.put(TemplateFillPlanner.LLM_INTERFACE_FIELD, BEDROCK);

        Result result = new Result();
        planner.fill(TEMPLATE_ID, "model-1", parameters, ActionListener.wrap(b -> result.body = b, e -> result.error = e));

        assertNotNull(result.error);
        verify(schemaResolver, never()).resolve(any(), any());
    }

    @Test
    public void fill_whenRequiredParamIsMissing_fails() {
        stubSchema(schemaWithQueryAndSize());
        stubToolUse(Map.of("size", 5));

        Result result = fill(params("cheap tents", BEDROCK));

        assertNotNull(result.error);
        assertTrue(result.error.getMessage().contains("required param 'lex_query'"));
        verify(renderer, never()).render(any(), any());
    }

    @Test
    public void fill_whenRenderFails_fails() {
        stubSchema(schemaWithQueryAndSize());
        stubToolUse(Map.of("lex_query", "tents"));
        when(renderer.render(any(), any())).thenThrow(new IllegalArgumentException("Template 'product_search' failed to render"));

        Result result = fill(params("cheap tents", BEDROCK));

        assertNotNull(result.error);
        assertTrue(result.error.getMessage().contains("failed to render"));
    }

    @Test
    public void fill_whenModelCallFails_fails() {
        stubSchema(schemaWithQueryAndSize());
        doAnswer((Answer<Void>) inv -> {
            ActionListener<MLTaskResponse> l = inv.getArgument(2);
            l.onFailure(new RuntimeException("throttled"));
            return null;
        }).when(client).execute(any(), any(), any());

        Result result = fill(params("cheap tents", BEDROCK));

        assertNotNull(result.error);
        assertEquals("throttled", result.error.getMessage());
    }

    @Test
    public void fill_whenToolCallIsEmpty_fails() {
        stubSchema(schemaWithQueryAndSize());
        stubToolUse(Map.of());

        Result result = fill(params("cheap tents", BEDROCK));

        assertNotNull(result.error);
        assertTrue(result.error.getMessage().contains("no input"));
    }

    @Test
    public void fill_whenModelReturnsNoTensors_fails() {
        stubSchema(schemaWithQueryAndSize());
        ModelTensorOutput empty = ModelTensorOutput.builder().mlModelOutputs(List.of()).build();
        stubPredictionResponse(empty);

        Result result = fill(params("cheap tents", BEDROCK));

        assertNotNull(result.error);
        assertTrue(result.error.getMessage().contains("no tensors"));
    }

    // ---- an unset interface must still force the tool ----------------------

    /**
     * A FLOW agent never merges the agent's own parameters into a tool's execute params, so
     * _llm_interface does not reach the tool unless an operator sets it on the tool spec.
     * Treating that silence as "unsupported" would quietly route every agentic-search request
     * to the weaker prompt-enforced fill.
     */
    @Test
    public void fill_withNoDeclaredInterface_stillForcesTheTool() {
        stubSchema(schemaWithQueryAndSize());
        stubToolUse(Map.of("lex_query", "tents"));
        when(renderer.render(any(), any())).thenReturn(Map.of("query", Map.of()));

        Result result = fill(params("cheap tents", null));

        assertNull(result.error);
        Map<String, String> sent = capturePredictionParams();
        assertNotNull(sent.get(ForcedToolCall.TOOL_CONFIG_PARAM));
        assertEquals(TemplateFillPromptTemplate.FILL_SYSTEM_PROMPT, sent.get(TemplateFillPlanner.SYSTEM_PROMPT_FIELD));
    }

    @Test
    public void fill_readsTheInterfaceFromThePlainerAlias() {
        stubSchema(schemaWithQueryAndSize());
        stubTextResponse("{\"params\":{\"lex_query\":\"tents\"}}");
        when(renderer.render(any(), any())).thenReturn(Map.of("query", Map.of()));

        Map<String, String> parameters = params("cheap tents", null);
        parameters.put(TemplateFillPlanner.LLM_INTERFACE_ALIAS_FIELD, "openai/v1/chat/completions");
        planner.fill(TEMPLATE_ID, "model-1", parameters, ActionListener.wrap(b -> {}, e -> {}));

        assertFalse(capturePredictionParams().containsKey(ForcedToolCall.TOOL_CONFIG_PARAM));
    }

    /**
     * Attempting the forced call costs nothing when the connector cannot honor it: the
     * toolConfig is simply never substituted, so the answer arrives as text and is read as
     * text rather than spending a second round trip.
     */
    @Test
    public void fill_whenForcedCallIsIgnored_readsTheTextResponseInstead() {
        stubSchema(schemaWithQueryAndSize());
        stubTextResponse("{\"params\":{\"lex_query\":\"tents\",\"size\":5}}");
        when(renderer.render(any(), any())).thenReturn(Map.of("query", Map.of()));

        Result result = fill(params("cheap tents", null));

        assertNull(result.error);
        @SuppressWarnings("unchecked")
        ArgumentCaptor<Map<String, Object>> renderParams = ArgumentCaptor.forClass(Map.class);
        verify(renderer).render(any(), renderParams.capture());
        assertEquals("tents", renderParams.getValue().get("lex_query"));
        assertEquals(5L, renderParams.getValue().get("size"));
    }

    @Test
    public void fill_whenForcedCallIsIgnoredAndTextIsProse_fails() {
        stubSchema(schemaWithQueryAndSize());
        stubTextResponse("I am not able to help with that.");

        Result result = fill(params("cheap tents", null));

        assertNotNull(result.error);
    }

    /** Abstention still lands when it comes back as text rather than a tool call. */
    @Test
    public void fill_whenForcedCallIsIgnoredAndTextAbstains_abstains() {
        stubSchema(schemaWithQueryAndSize());
        stubTextResponse("{\"params\":{},\"cannot_express\":true}");

        Result result = fill(params("how many per brand?", null));

        assertTrue(result.error instanceof TemplateFillPlanner.TemplateCannotExpressException);
        verify(renderer, never()).render(any(), any());
    }

    // ---- the prompt-enforced path -----------------------------------------

    /** Without a forcible interface the fill still runs, carried by the prompt instead. */
    @Test
    public void fill_withoutForcedToolSupport_usesPromptEnforcedJson() {
        stubSchema(schemaWithQueryAndSize());
        stubTextResponse("Here you go: {\"params\":{\"lex_query\":\"tents\",\"size\":5},\"cannot_express\":false}");
        when(renderer.render(any(), any())).thenReturn(Map.of("query", Map.of()));

        Result result = fill(params("cheap tents", UNFORCEABLE));

        assertNull(result.error);
        Map<String, String> sent = capturePredictionParams();
        assertFalse(sent.containsKey(ForcedToolCall.TOOL_CONFIG_PARAM));
        assertEquals(TemplateFillPromptTemplate.FILL_SYSTEM_PROMPT_JSON_FALLBACK, sent.get(TemplateFillPlanner.SYSTEM_PROMPT_FIELD));
        // The schema has to travel in the prompt, since no tool schema carries it.
        assertTrue(sent.get(TemplateFillPlanner.FILL_SCHEMA_FIELD).contains("lex_query"));

        @SuppressWarnings("unchecked")
        ArgumentCaptor<Map<String, Object>> renderParams = ArgumentCaptor.forClass(Map.class);
        verify(renderer).render(any(), renderParams.capture());
        assertEquals("tents", renderParams.getValue().get("lex_query"));
        assertEquals(5L, renderParams.getValue().get("size"));
    }

    @Test
    public void fill_promptEnforcedPathHonorsAbstention() {
        stubSchema(schemaWithQueryAndSize());
        stubTextResponse("{\"params\":{},\"cannot_express\":true}");

        Result result = fill(params("how many per brand?", UNFORCEABLE));

        assertTrue(result.error instanceof TemplateFillPlanner.TemplateCannotExpressException);
        verify(renderer, never()).render(any(), any());
    }

    /** A bare object with no params envelope is still usable. */
    @Test
    public void fill_promptEnforcedPathAcceptsABareFill() {
        stubSchema(schemaWithQueryAndSize());
        stubTextResponse("{\"lex_query\":\"tents\"}");
        when(renderer.render(any(), any())).thenReturn(Map.of("query", Map.of()));

        Result result = fill(params("cheap tents", UNFORCEABLE));

        assertNull(result.error);
    }

    @Test
    public void fill_promptEnforcedPathFailsOnUnparseableText() {
        stubSchema(schemaWithQueryAndSize());
        stubTextResponse("I could not do that.");

        Result result = fill(params("cheap tents", UNFORCEABLE));

        assertNotNull(result.error);
    }

    // ---- isTrue ------------------------------------------------------------

    @Test
    public void isTrue_readsTheFlagWithoutPlainTruthiness() {
        assertTrue(TemplateFillPlanner.isTrue(true));
        assertTrue(TemplateFillPlanner.isTrue("true"));
        assertTrue(TemplateFillPlanner.isTrue("TRUE"));
        assertTrue(TemplateFillPlanner.isTrue("yes"));
        assertTrue(TemplateFillPlanner.isTrue("1"));
        assertTrue(TemplateFillPlanner.isTrue(1));
        assertFalse(TemplateFillPlanner.isTrue(false));
        assertFalse(TemplateFillPlanner.isTrue("false"));
        assertFalse(TemplateFillPlanner.isTrue("no"));
        assertFalse(TemplateFillPlanner.isTrue(0));
        assertFalse(TemplateFillPlanner.isTrue(null));
    }

    // ---- helpers -----------------------------------------------------------

    private static Map<String, Object> schemaWithQueryAndSize() {
        Map<String, Object> schema = new LinkedHashMap<>();
        schema.put("lex_query", Map.of("type", "string", "required", true, "description", "content words only"));
        schema.put("size", Map.of("type", "number", "description", "how many hits"));
        return schema;
    }

    private Map<String, String> params(String question, String llmInterface) {
        Map<String, String> parameters = new HashMap<>();
        parameters.put(TemplateFillPlanner.QUESTION_FIELD, question);
        if (llmInterface != null) {
            parameters.put(TemplateFillPlanner.LLM_INTERFACE_FIELD, llmInterface);
        }
        return parameters;
    }

    private Result fill(Map<String, String> parameters) {
        Result result = new Result();
        planner.fill(TEMPLATE_ID, "model-1", parameters, ActionListener.wrap(b -> result.body = b, e -> result.error = e));
        return result;
    }

    private void stubSchema(Map<String, Object> paramSchema) {
        AgenticSearchTemplate template = AgenticSearchTemplate
            .builder()
            .templateId(TEMPLATE_ID)
            .indexBinding("products")
            .paramSchema(paramSchema)
            .build();
        doAnswer((Answer<Void>) inv -> {
            ActionListener<AgenticSearchTemplate> l = inv.getArgument(1);
            l.onResponse(template);
            return null;
        }).when(schemaResolver).resolve(any(), any());
    }

    /** Shape a Bedrock Converse tool-use response around the given arguments. */
    private void stubToolUse(Map<String, Object> arguments) {
        Map<String, Object> toolUse = Map.of("toolUse", Map.of("name", FillToolSchemaBuilder.FILL_TOOL_NAME, "input", arguments));
        Map<String, Object> data = Map.of("output", Map.of("message", Map.of("content", List.of(toolUse))));
        stubPredictionResponse(tensorOutput(data));
    }

    private void stubTextResponse(String text) {
        stubPredictionResponse(tensorOutput(Map.of("response", text)));
    }

    private static ModelTensorOutput tensorOutput(Map<String, Object> dataAsMap) {
        ModelTensor tensor = ModelTensor.builder().name("response").dataAsMap(dataAsMap).build();
        ModelTensors tensors = ModelTensors.builder().mlModelTensors(List.of(tensor)).build();
        return ModelTensorOutput.builder().mlModelOutputs(List.of(tensors)).build();
    }

    private void stubPredictionResponse(ModelTensorOutput output) {
        MLTaskResponse response = MLTaskResponse.builder().output(output).build();
        doAnswer((Answer<Void>) inv -> {
            ActionListener<MLTaskResponse> l = inv.getArgument(2);
            l.onResponse(response);
            return null;
        }).when(client).execute(any(), any(), any());
    }

    @SuppressWarnings("unchecked")
    private Map<String, String> capturePredictionParams() {
        ArgumentCaptor<MLPredictionTaskRequest> captor = ArgumentCaptor.forClass(MLPredictionTaskRequest.class);
        verify(client, times(1)).execute(org.mockito.ArgumentMatchers.eq(MLPredictionTaskAction.INSTANCE), captor.capture(), any());
        return ((org.opensearch.ml.common.dataset.remote.RemoteInferenceInputDataSet) captor.getValue().getMlInput().getInputDataset())
            .getParameters();
    }

    private static final class Result {
        private Map<String, Object> body;
        private Exception error;
    }

    /** Guards against the constant drifting out of sync with the vocabulary the runners use. */
    @Test
    public void llmInterfaceConstantMatchesTheAgentVocabulary() {
        assertTrue(ForcedToolCall.supportsForcedTool("bedrock/converse/claude"));
        assertTrue(ForcedToolCall.supportsForcedTool("bedrock/converse"));
        assertTrue(ForcedToolCall.supportsForcedTool("BEDROCK/CONVERSE/CLAUDE"));
        assertFalse(ForcedToolCall.supportsForcedTool("openai/v1/chat/completions"));
        assertFalse(ForcedToolCall.supportsForcedTool(null));
        assertFalse(ForcedToolCall.supportsForcedTool("  "));
        assertNull(ForcedToolCall.resultPath("openai/v1/chat/completions"));
        assertEquals(ForcedToolCall.BEDROCK_TOOL_USE_RESULT_PATH, ForcedToolCall.resultPath("bedrock/converse/claude"));
        assertNotNull(StringUtils.gson);
    }
}
