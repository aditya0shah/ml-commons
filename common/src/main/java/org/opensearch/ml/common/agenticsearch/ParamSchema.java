/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.common.agenticsearch;

import java.util.List;
import java.util.Map;

/**
 * The {@code param_schema} entry vocabulary and the value rules that go with it,
 * shared by the register-time deriver and the query-time filler.
 *
 * <p>These two halves live in different modules — derivation and validation in
 * {@code plugin}, the fill/render path in {@code ml-algorithms}, which cannot depend on
 * {@code plugin} — so the vocabulary sits in {@code common} where both reach it. Keeping
 * one copy matters: if the two sides disagree about what a type accepts, a schema can
 * pass registration and then produce values that will not render, and the fill path
 * absorbs render failures into its fallback, so the disagreement would surface only as
 * added latency.
 *
 * <p>An entry is {@code {type, required?, enum?, description?, source?}}. Only
 * {@code type} is mandatory; a missing {@code required} means {@code false}.
 */
public final class ParamSchema {

    // Entry keys.
    public static final String TYPE_KEY = "type";
    public static final String REQUIRED_KEY = "required";
    public static final String DESCRIPTION_KEY = "description";
    public static final String ENUM_KEY = "enum";
    public static final String SOURCE_KEY = "source";

    // Structural types inferable from a Mustache body alone.
    public static final String TYPE_STRING = "string";
    public static final String TYPE_NUMBER = "number";
    public static final String TYPE_BOOLEAN = "boolean";
    public static final String TYPE_ARRAY = "array";

    /** Value of {@link #SOURCE_KEY} for an enum derived from the index mapping. */
    public static final String SOURCE_MAPPING = "mapping";

    private ParamSchema() {}

    /**
     * Whether a value is usable for a param of {@code type}.
     *
     * <p>An {@code array} param is a triple-stache slot carrying raw JSON, so its value
     * must be a {@code String}. A {@code List} is wrong twice over: the Mustache engine
     * stringifies it ({@code [{"term":{"t":"a"}}]} renders as {@code {0={term={t=a}}}}),
     * and because the value also guards its own section a multi-element list makes that
     * section iterate, emitting the clause once per element. Both produce invalid JSON
     * that fails only at query time.
     */
    public static boolean valueFitsType(Object value, String type) {
        if (value == null) {
            return false;
        }
        switch (type) {
            case TYPE_NUMBER:
                return value instanceof Number;
            case TYPE_BOOLEAN:
                return value instanceof Boolean;
            case TYPE_ARRAY:
                return value instanceof String;
            default:
                return value instanceof String;
        }
    }

    /** The declared type of a param spec, or {@link #TYPE_STRING} when absent. */
    public static String typeOf(Map<String, Object> spec) {
        Object type = spec.get(TYPE_KEY);
        return (type instanceof String) && !((String) type).isEmpty() ? (String) type : TYPE_STRING;
    }

    /** Whether a param spec is required. An absent {@code required} means {@code false}. */
    public static boolean isRequired(Map<String, Object> spec) {
        return Boolean.TRUE.equals(spec.get(REQUIRED_KEY));
    }

    /** The param's description, or {@code ""} when absent. */
    public static String descriptionOf(Map<String, Object> spec) {
        Object description = spec.get(DESCRIPTION_KEY);
        return description instanceof String ? (String) description : "";
    }

    /**
     * The param's enum values, or {@code null} when it has none. An empty or non-list
     * {@code enum} is treated as absent here; {@code validateParamSchema} rejects it at
     * registration, and the query path must not fail a whole fill over one bad entry.
     */
    public static List<?> enumOf(Map<String, Object> spec) {
        Object enumValues = spec.get(ENUM_KEY);
        if (enumValues instanceof List && !((List<?>) enumValues).isEmpty()) {
            return (List<?>) enumValues;
        }
        return null;
    }

    /**
     * The spec map for one param, or {@code null} if the entry is not an object.
     * A malformed entry is skipped rather than fatal on the query path.
     */
    @SuppressWarnings("unchecked")
    public static Map<String, Object> specOf(Object entry) {
        return entry instanceof Map ? (Map<String, Object>) entry : null;
    }
}
