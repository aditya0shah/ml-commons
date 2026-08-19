/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.engine.tools.templatefill;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

import org.opensearch.ml.common.agenticsearch.ParamSchema;

/**
 * Turns a stored param-schema into the JSON Schema for the tool the model is forced to
 * call. The schema is the whole guidance: the fill prompt carries no mapping, no examples
 * and no DSL rules, so per-param descriptions and enums are what steer the fill.
 *
 * <p>Property names are the real Mustache param names, dots and all, because those are
 * the keys the render needs back. Only params marked {@code required} land in the schema's
 * {@code required} list; an unset optional is then simply absent from the fill, which is
 * what lets the body's inverted-section defaults ({@code {{^size}}10{{/size}}}) and
 * optional clauses ({@code {{#color}}...{{/color}}}) behave as authored. Emitting an
 * explicit null instead would render an empty slot and break the JSON.
 *
 * <p>Optionals are declared with their plain type rather than a nullable union. The agent
 * server emits {@code ["integer","null"]} there, but that is an artifact of how Pydantic
 * renders {@code Optional[int]}, not a requirement: absence from {@code required} already
 * says the param may be omitted, and a plain type gives the model no reason to emit null.
 */
public final class FillToolSchemaBuilder {

    /** The tool the model is forced to call on the single-template path. */
    public static final String FILL_TOOL_NAME = "FillTemplate";
    public static final String FILL_TOOL_DESCRIPTION = "Fill the search template's parameters for this question.";

    /** The synthetic abstain flag. Never a Mustache param, never stored, never rendered. */
    public static final String CANNOT_EXPRESS_FIELD = "cannot_express";

    /**
     * The abstain trigger list is enumerated rather than abstract on purpose. Vaguer or
     * reasoning-heavy wordings measurably over-abstained in a prompt sweep, dropping to the
     * slower fallback on questions the template could actually serve. Change this with a
     * benchmark re-run, not by feel.
     */
    public static final String CANNOT_EXPRESS_DESCRIPTION = "Set true when the question needs a capability NOT among these "
        + "parameters. Concretely set it true if the question: (a) restricts text matching to ONE specific field "
        + "(e.g. 'in the title', 'in the name') and no parameter isolates that field; (b) demands an EXACT contiguous "
        + "phrase / literal wording in a field and no phrase parameter exists; (c) asks to RANK or BOOST by a signal "
        + "(most popular, trending, boost recent/newer, custom relevance) and no parameter or sort option expresses that "
        + "ranking; (d) asks for a COUNT-only answer, aggregation, faceting, or grouping; (e) references a field, "
        + "similarity ('products like X'), or predicate that has no matching parameter. Otherwise leave it false and "
        + "fill the parameters.";

    /** Appended to an array param's description; the slot renders raw through a triple brace. */
    static final String ARRAY_HINT = " JSON array literal, e.g. [\"a\",\"b\"].";

    // JSON Schema type names. The deriver only ever emits ParamSchema's four structural
    // types, but a caller-registered schema may use the narrower numeric names.
    private static final Map<String, String> JSON_TYPES = Map
        .ofEntries(
            Map.entry("string", "string"),
            Map.entry("text", "string"),
            Map.entry("keyword", "string"),
            Map.entry("integer", "integer"),
            Map.entry("int", "integer"),
            Map.entry("long", "integer"),
            Map.entry("number", "number"),
            Map.entry("float", "number"),
            Map.entry("double", "number"),
            Map.entry("boolean", "boolean"),
            Map.entry("bool", "boolean"),
            // An array param is a raw-JSON slot, so it travels as a string holding a JSON
            // array literal. A real JSON array would be stringified by the Mustache engine.
            Map.entry("array", "string")
        );

    private FillToolSchemaBuilder() {}

    /**
     * Whether the template declares a real param named {@code cannot_express}. The real
     * param wins and the abstain hatch is disabled for that template, rather than one
     * shadowing the other.
     */
    public static boolean abstainDisabled(Map<String, Object> paramSchema) {
        return paramSchema != null && paramSchema.containsKey(CANNOT_EXPRESS_FIELD);
    }

    /**
     * Build the tool's {@code inputSchema} JSON for one template.
     *
     * @param paramSchema the stored param-schema
     * @return a JSON Schema object; params in schema order, with the abstain flag appended last
     */
    public static Map<String, Object> buildInputSchema(Map<String, Object> paramSchema) {
        if (paramSchema == null || paramSchema.isEmpty()) {
            throw new IllegalArgumentException("param-schema is empty; nothing to fill");
        }
        Map<String, Object> properties = new LinkedHashMap<>();
        List<String> required = new ArrayList<>();
        for (Map.Entry<String, Object> entry : paramSchema.entrySet()) {
            Map<String, Object> spec = ParamSchema.specOf(entry.getValue());
            if (spec == null) {
                // A malformed entry is skipped rather than failing the whole fill; the
                // render-parse guard still backstops whatever comes back.
                continue;
            }
            properties.put(entry.getKey(), paramJsonSchema(spec));
            if (ParamSchema.isRequired(spec)) {
                required.add(entry.getKey());
            }
        }
        if (properties.isEmpty()) {
            throw new IllegalArgumentException("param-schema declares no usable params");
        }
        if (!abstainDisabled(paramSchema)) {
            properties.put(CANNOT_EXPRESS_FIELD, abstainProperty());
        }

        Map<String, Object> schema = new LinkedHashMap<>();
        schema.put("type", "object");
        schema.put("properties", properties);
        schema.put("required", required);
        return schema;
    }

    static Map<String, Object> abstainProperty() {
        Map<String, Object> property = new LinkedHashMap<>();
        property.put("type", "boolean");
        property.put("default", false);
        property.put("description", CANNOT_EXPRESS_DESCRIPTION);
        return property;
    }

    /** The JSON Schema for one param. */
    static Map<String, Object> paramJsonSchema(Map<String, Object> spec) {
        String declaredType = ParamSchema.typeOf(spec);
        String jsonType = JSON_TYPES.getOrDefault(declaredType.toLowerCase(Locale.ROOT), "string");
        String description = ParamSchema.descriptionOf(spec);
        if (ParamSchema.TYPE_ARRAY.equalsIgnoreCase(declaredType)) {
            description = description.isEmpty() ? ARRAY_HINT.trim() : description + ARRAY_HINT;
        }

        Map<String, Object> property = new LinkedHashMap<>();
        property.put("type", jsonType);
        if (!description.isEmpty()) {
            property.put("description", description);
        }

        List<?> enumValues = ParamSchema.enumOf(spec);
        if (enumValues != null) {
            // Keep the declared type alongside the enum: a numeric enum declared as a
            // string would have the model emit "5", which then fails the value check.
            // When the members are not all strings but the type resolved to string, the
            // type is dropped and the enum alone constrains the slot.
            boolean allStrings = enumValues.stream().allMatch(v -> v instanceof String);
            if (!allStrings && "string".equals(jsonType)) {
                property.remove("type");
            }
            property.put("enum", enumValues);
        }
        return property;
    }
}
