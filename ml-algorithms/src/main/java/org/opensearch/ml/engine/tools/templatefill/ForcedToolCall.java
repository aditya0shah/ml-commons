/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.engine.tools.templatefill;

import static org.opensearch.ml.engine.algorithms.agent.AgentUtils.LLM_INTERFACE_BEDROCK_CONVERSE;
import static org.opensearch.ml.engine.algorithms.agent.AgentUtils.LLM_INTERFACE_BEDROCK_CONVERSE_CLAUDE;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

import org.opensearch.ml.common.utils.StringUtils;

/**
 * Forces the model to call one tool with a caller-supplied JSON Schema, which is what
 * makes a fill both cheap and structurally reliable: the model emits values only, with no
 * room for a natural-language preamble.
 *
 * <p>The mechanism is the connector's structured-output injection, not the agent
 * function-calling layer. That layer only ever sends {@code toolChoice} implicitly as AUTO
 * — there is no "required" or specific-tool option in it. What does exist is
 * {@code HttpConnector.injectStructuredOutputParams}: a prediction parameter named
 * {@code _toolConfig_json} replaces the request body's top-level {@code toolConfig} field,
 * provided the connector's PREDICT action sets {@code supports_structured_output: true}.
 * The memory fact-extraction path already does exactly this, including the forced
 * {@code toolChoice}, so this is a second caller of a proven mechanism rather than new
 * plumbing.
 *
 * <p>Only Bedrock Converse is wired up here. It is the interface the feature was measured
 * on, and the other providers express constrained decoding differently enough (a response
 * format or a generation config rather than a forced tool) that guessing at them adds
 * surface without evidence. Anything else degrades to a prompt-enforced JSON fill, which
 * still fills templates, just without the schema-level guarantee.
 *
 * <p>Two things a caller must honor. The connector needs
 * {@code supports_structured_output: true} or the injection is silently a no-op and the
 * model answers in prose. And {@code _structured_output_result_path} is for local use: it
 * says where in the model's response the tool arguments land, and must not be forwarded to
 * the connector as a body parameter.
 */
public final class ForcedToolCall {

    /** Prediction parameter that replaces the request body's toolConfig field. */
    public static final String TOOL_CONFIG_PARAM = "_toolConfig_json";

    /** Where Bedrock Converse puts a tool call's arguments. */
    public static final String BEDROCK_TOOL_USE_RESULT_PATH = "$.output.message.content[0].toolUse.input";

    private ForcedToolCall() {}

    /**
     * Whether this LLM interface can be made to call a specific tool with a supplied schema.
     * Uses the same {@code _llm_interface} vocabulary the agent runners already thread through.
     */
    public static boolean supportsForcedTool(String llmInterface) {
        if (llmInterface == null || llmInterface.isBlank()) {
            return false;
        }
        String normalized = llmInterface.trim().toLowerCase(Locale.ROOT);
        return LLM_INTERFACE_BEDROCK_CONVERSE_CLAUDE.equals(normalized) || LLM_INTERFACE_BEDROCK_CONVERSE.equals(normalized);
    }

    /**
     * The JSON path at which the forced tool's arguments arrive, or null when this
     * interface cannot force a tool.
     */
    public static String resultPath(String llmInterface) {
        return supportsForcedTool(llmInterface) ? BEDROCK_TOOL_USE_RESULT_PATH : null;
    }

    /**
     * Build the {@code _toolConfig_json} value: one tool, and a {@code toolChoice} naming
     * it. Exactly one tool is declared so the model has nothing else to choose.
     *
     * @param toolName the tool to force
     * @param description shown to the model alongside the schema
     * @param inputSchema the JSON Schema for the tool's arguments
     */
    public static String toolConfigJson(String toolName, String description, Map<String, Object> inputSchema) {
        Map<String, Object> toolSpec = new LinkedHashMap<>();
        toolSpec.put("name", toolName);
        toolSpec.put("description", description);
        toolSpec.put("inputSchema", Map.of("json", inputSchema));

        Map<String, Object> toolConfig = new LinkedHashMap<>();
        toolConfig.put("tools", List.of(Map.of("toolSpec", toolSpec)));
        toolConfig.put("toolChoice", Map.of("tool", Map.of("name", toolName)));
        return StringUtils.gson.toJson(toolConfig);
    }
}
