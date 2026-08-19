/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.engine.tools.templatefill;

/**
 * Prompts for the template-fill path.
 *
 * <p>These are short on purpose. The fill prompt carries no index mapping, no sample
 * document, and no DSL rules, because the tool schema already carries the per-param
 * descriptions and enums that steer the fill — that omission is where the output-token
 * saving comes from, and the saving is the point of the whole path.
 *
 * <p>The wording is ported verbatim from the implementation these prompts were benchmarked
 * against. Both the abstain instruction here and the trigger enumeration in
 * {@link FillToolSchemaBuilder#CANNOT_EXPRESS_DESCRIPTION} were tuned by a prompt sweep, in
 * which vaguer or heavier phrasings over-abstained and lost accuracy on questions the
 * template could serve. Change them with a benchmark re-run, not by feel.
 */
public final class TemplateFillPromptTemplate {

    private TemplateFillPromptTemplate() {}

    public static final String FILL_SYSTEM_PROMPT =
        "You extract search parameters from a user's question to fill a predefined OpenSearch search template.\n"
            + "Call the FillTemplate tool exactly once. Fill only the parameters the question clearly implies; leave everything else unset — do not guess.\n"
            + "Put ONLY content/topic words in any free-text query parameter — never counts, filters, sort terms, or field names.\n"
            + "For enum parameters, choose only from the options that parameter allows.\n"
            + "If the question needs something these parameters cannot express — a field not listed here, prefix/wildcard/fuzzy matching, aggregations, an unsupported range, or custom scoring — set cannot_express=true and leave the other parameters unset. Do not force an approximate fill; abstaining routes the question to a more capable path.";

    public static final String FILL_USER_PROMPT = "Question: ${parameters.question}\n\n"
        + "Fill the FillTemplate tool's parameters for this question.\n";

    /**
     * Used when the model's interface cannot force a tool call. The instruction has to
     * carry the contract the tool schema would otherwise impose, so it names the envelope
     * explicitly and repeats that unset params must be omitted rather than nulled.
     */
    public static final String FILL_SYSTEM_PROMPT_JSON_FALLBACK =
        "You extract search parameters from a user's question to fill a predefined OpenSearch search template.\n"
            + "Fill only the parameters the question clearly implies; leave everything else unset — do not guess.\n"
            + "Put ONLY content/topic words in any free-text query parameter — never counts, filters, sort terms, or field names.\n"
            + "For enum parameters, choose only from the options that parameter allows.\n"
            + "If the question needs something these parameters cannot express — a field not listed here, prefix/wildcard/fuzzy matching, aggregations, an unsupported range, or custom scoring — set cannot_express to true and fill no parameters. Do not force an approximate fill; abstaining routes the question to a more capable path.\n"
            + "Respond with ONLY a JSON object of this shape, and nothing else:\n"
            + "{\"params\": {\"<param name>\": <value>}, \"cannot_express\": false}\n"
            + "Omit any parameter you are not filling; do not include it with a null value.\n"
            + "These are the parameters, as a JSON Schema:\n"
            + "${parameters.fill_schema}";

    public static final String FILL_USER_PROMPT_JSON_FALLBACK = "Question: ${parameters.question}\n\n"
        + "Emit the JSON object filling this template's parameters for this question.\n";
}
