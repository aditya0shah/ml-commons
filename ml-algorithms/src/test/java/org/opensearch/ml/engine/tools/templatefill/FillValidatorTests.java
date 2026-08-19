/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.engine.tools.templatefill;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import org.junit.Test;

public class FillValidatorTests {

    private static Map<String, Object> spec(Object... kv) {
        Map<String, Object> m = new LinkedHashMap<>();
        for (int i = 0; i < kv.length; i += 2) {
            m.put((String) kv[i], kv[i + 1]);
        }
        return m;
    }

    private static Map<String, Object> schema(Object... kv) {
        Map<String, Object> m = new LinkedHashMap<>();
        for (int i = 0; i < kv.length; i += 2) {
            m.put((String) kv[i], kv[i + 1]);
        }
        return m;
    }

    // ---- the integral-number rule ------------------------------------------

    /**
     * The rule the whole coercion layer exists for: a JSON parser hands back 5 as a
     * double, and a double renders as "5.0", which OpenSearch rejects for size.
     */
    @Test
    public void validate_keepsIntegralNumbersIntegral() {
        Map<String, Object> paramSchema = schema("size", spec("type", "number"));

        Map<String, Object> out = FillValidator.validate(paramSchema, Map.of("size", 5.0d));

        assertEquals(5L, out.get("size"));
        assertEquals("5", String.valueOf(out.get("size")));
    }

    @Test
    public void validate_preservesGenuineFractions() {
        Map<String, Object> paramSchema = schema("boost", spec("type", "number"));

        Map<String, Object> out = FillValidator.validate(paramSchema, Map.of("boost", 1.5d));

        assertEquals(1.5d, out.get("boost"));
    }

    @Test
    public void validate_floatTypeStaysFractional() {
        Map<String, Object> paramSchema = schema("boost", spec("type", "float"));

        Map<String, Object> out = FillValidator.validate(paramSchema, Map.of("boost", 2));

        assertEquals(2.0d, out.get("boost"));
    }

    @Test
    public void validate_integerTypeRejectsAFraction() {
        Map<String, Object> paramSchema = schema("size", spec("type", "integer"));

        IllegalArgumentException e = assertThrows(
            IllegalArgumentException.class,
            () -> FillValidator.validate(paramSchema, Map.of("size", 2.5d))
        );
        assertTrue(e.getMessage().contains("expects an integer"));
    }

    @Test
    public void validate_acceptsAQuotedNumber() {
        Map<String, Object> paramSchema = schema("size", spec("type", "number"));

        Map<String, Object> out = FillValidator.validate(paramSchema, Map.of("size", "10"));

        assertEquals(10L, out.get("size"));
    }

    @Test
    public void validate_rejectsNonNumericTextForANumber() {
        Map<String, Object> paramSchema = schema("size", spec("type", "number"));

        assertThrows(IllegalArgumentException.class, () -> FillValidator.validate(paramSchema, Map.of("size", "ten")));
    }

    // ---- required / optional ------------------------------------------------

    @Test
    public void validate_failsWhenARequiredParamIsMissing() {
        Map<String, Object> paramSchema = schema("lex_query", spec("type", "string", "required", true));

        IllegalArgumentException e = assertThrows(IllegalArgumentException.class, () -> FillValidator.validate(paramSchema, Map.of()));
        assertTrue(e.getMessage().contains("required param 'lex_query' was not filled"));
    }

    /** Omitted, never null: a null would render an empty slot and break the body's JSON. */
    @Test
    public void validate_omitsUnsetOptionals() {
        Map<String, Object> paramSchema = schema(
            "lex_query",
            spec("type", "string", "required", true),
            "size",
            spec("type", "number"),
            "color",
            spec("type", "string")
        );

        Map<String, Object> out = FillValidator.validate(paramSchema, Map.of("lex_query", "tents"));

        assertEquals(Map.of("lex_query", "tents"), out);
        assertFalse(out.containsKey("size"));
        assertFalse(out.containsKey("color"));
    }

    @Test
    public void validate_treatsAnExplicitNullAsUnset() {
        Map<String, Object> paramSchema = schema("size", spec("type", "number"));
        Map<String, Object> emitted = new LinkedHashMap<>();
        emitted.put("size", null);

        Map<String, Object> out = FillValidator.validate(paramSchema, emitted);

        assertTrue(out.isEmpty());
    }

    @Test
    public void validate_failsWhenARequiredParamIsExplicitlyNull() {
        Map<String, Object> paramSchema = schema("lex_query", spec("type", "string", "required", true));
        Map<String, Object> emitted = new LinkedHashMap<>();
        emitted.put("lex_query", null);

        assertThrows(IllegalArgumentException.class, () -> FillValidator.validate(paramSchema, emitted));
    }

    /** A caller-registered schema may omit required; absent means not required. */
    @Test
    public void validate_absentRequiredMeansOptional() {
        Map<String, Object> paramSchema = schema("size", spec("type", "number"));

        assertTrue(FillValidator.validate(paramSchema, Map.of()).isEmpty());
    }

    // ---- unknown keys ------------------------------------------------------

    /** One hallucinated key must not cost the whole fill. */
    @Test
    public void validate_dropsUnknownKeys() {
        Map<String, Object> paramSchema = schema("lex_query", spec("type", "string", "required", true));

        Map<String, Object> out = FillValidator.validate(paramSchema, Map.of("lex_query", "tents", "invented", "nonsense"));

        assertEquals(Map.of("lex_query", "tents"), out);
    }

    /** The abstain flag is not a template param, so it must never reach the renderer. */
    @Test
    public void validate_dropsTheAbstainFlag() {
        Map<String, Object> paramSchema = schema("lex_query", spec("type", "string", "required", true));

        Map<String, Object> out = FillValidator
            .validate(paramSchema, Map.of("lex_query", "tents", FillToolSchemaBuilder.CANNOT_EXPRESS_FIELD, false));

        assertEquals(Map.of("lex_query", "tents"), out);
    }

    @Test
    public void validate_skipsMalformedSchemaEntries() {
        Map<String, Object> paramSchema = schema("lex_query", spec("type", "string"), "broken", "not-an-object");

        Map<String, Object> out = FillValidator.validate(paramSchema, Map.of("lex_query", "tents", "broken", "x"));

        assertEquals(Map.of("lex_query", "tents"), out);
    }

    // ---- enums -------------------------------------------------------------

    @Test
    public void validate_acceptsAnEnumMember() {
        Map<String, Object> paramSchema = schema("sort_order", spec("type", "string", "enum", List.of("asc", "desc")));

        Map<String, Object> out = FillValidator.validate(paramSchema, Map.of("sort_order", "desc"));

        assertEquals("desc", out.get("sort_order"));
    }

    @Test
    public void validate_rejectsAValueOutsideTheEnum() {
        Map<String, Object> paramSchema = schema("sort_order", spec("type", "string", "enum", List.of("asc", "desc")));

        IllegalArgumentException e = assertThrows(
            IllegalArgumentException.class,
            () -> FillValidator.validate(paramSchema, Map.of("sort_order", "sideways"))
        );
        assertTrue(e.getMessage().contains("is not one of"));
    }

    /** The stored doc parses 5 as an Integer while the fill coerces to a Long. */
    @Test
    public void validate_matchesNumericEnumsAcrossBoxes() {
        Map<String, Object> paramSchema = schema("size", spec("type", "number", "enum", List.of(1, 5, 10)));

        Map<String, Object> out = FillValidator.validate(paramSchema, Map.of("size", 5.0d));

        assertEquals(5L, out.get("size"));
    }

    @Test
    public void validate_enumRejectionSurvivesCoercion() {
        Map<String, Object> paramSchema = schema("size", spec("type", "number", "enum", List.of(1, 5, 10)));

        assertThrows(IllegalArgumentException.class, () -> FillValidator.validate(paramSchema, Map.of("size", 7)));
    }

    @Test
    public void validate_ignoresAnEmptyEnum() {
        Map<String, Object> paramSchema = schema("sort_order", spec("type", "string", "enum", new ArrayList<>()));

        Map<String, Object> out = FillValidator.validate(paramSchema, Map.of("sort_order", "anything"));

        assertEquals("anything", out.get("sort_order"));
    }

    // ---- booleans and arrays ----------------------------------------------

    @Test
    public void validate_acceptsABoolean() {
        Map<String, Object> paramSchema = schema("sem_enabled", spec("type", "boolean"));

        assertEquals(true, FillValidator.validate(paramSchema, Map.of("sem_enabled", true)).get("sem_enabled"));
    }

    @Test
    public void validate_coercesAQuotedBoolean() {
        Map<String, Object> paramSchema = schema("sem_enabled", spec("type", "boolean"));

        assertEquals(true, FillValidator.validate(paramSchema, Map.of("sem_enabled", "true")).get("sem_enabled"));
        assertEquals(false, FillValidator.validate(paramSchema, Map.of("sem_enabled", "False")).get("sem_enabled"));
    }

    /** 1 is not true here: it would render as 1 and OpenSearch rejects that for a boolean field. */
    @Test
    public void validate_rejectsANumberForABoolean() {
        Map<String, Object> paramSchema = schema("sem_enabled", spec("type", "boolean"));

        assertThrows(IllegalArgumentException.class, () -> FillValidator.validate(paramSchema, Map.of("sem_enabled", 1)));
    }

    @Test
    public void validate_passesAnArrayParamThroughAsRawJson() {
        Map<String, Object> paramSchema = schema("filters", spec("type", "array"));

        Map<String, Object> out = FillValidator.validate(paramSchema, Map.of("filters", "[{\"term\":{\"color\":\"red\"}}]"));

        assertEquals("[{\"term\":{\"color\":\"red\"}}]", out.get("filters"));
    }

    /**
     * A real list would be stringified into Java map notation by the Mustache engine, and
     * because the value guards its own section a multi-element list would make it iterate.
     */
    @Test
    public void validate_serializesAListForAnArrayParam() {
        Map<String, Object> paramSchema = schema("filters", spec("type", "array"));

        Map<String, Object> out = FillValidator.validate(paramSchema, Map.of("filters", List.of("a", "b")));

        assertEquals("[\"a\",\"b\"]", out.get("filters"));
        assertTrue(out.get("filters") instanceof String);
    }

    // ---- strings and ordering ---------------------------------------------

    @Test
    public void validate_coercesAScalarToStringForAStringParam() {
        Map<String, Object> paramSchema = schema("lex_query", spec("type", "string"));

        assertEquals("5", FillValidator.validate(paramSchema, Map.of("lex_query", 5)).get("lex_query"));
    }

    @Test
    public void validate_treatsAnUnknownTypeAsString() {
        Map<String, Object> paramSchema = schema("where", spec("type", "geo_point"));

        assertEquals("here", FillValidator.validate(paramSchema, Map.of("where", "here")).get("where"));
    }

    @Test
    public void validate_returnsParamsInSchemaOrder() {
        Map<String, Object> paramSchema = schema(
            "lex_query",
            spec("type", "string"),
            "size",
            spec("type", "number"),
            "sort_order",
            spec("type", "string")
        );

        Map<String, Object> out = FillValidator.validate(paramSchema, Map.of("sort_order", "asc", "size", 5, "lex_query", "tents"));

        assertEquals(List.of("lex_query", "size", "sort_order"), new ArrayList<>(out.keySet()));
    }

    @Test
    public void validate_rejectsAnEmptySchema() {
        assertThrows(IllegalArgumentException.class, () -> FillValidator.validate(Map.of(), Map.of()));
    }

    @Test
    public void validate_toleratesANullFill() {
        Map<String, Object> paramSchema = schema("size", spec("type", "number"));

        assertTrue(FillValidator.validate(paramSchema, null).isEmpty());
    }
}
