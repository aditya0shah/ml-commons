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

public class FillToolSchemaBuilderTests {

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

    @SuppressWarnings("unchecked")
    private static Map<String, Object> properties(Map<String, Object> inputSchema) {
        return (Map<String, Object>) inputSchema.get("properties");
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> property(Map<String, Object> inputSchema, String name) {
        return (Map<String, Object>) properties(inputSchema).get(name);
    }

    @SuppressWarnings("unchecked")
    private static List<String> required(Map<String, Object> inputSchema) {
        return (List<String>) inputSchema.get("required");
    }

    @Test
    public void build_producesAnObjectSchema() {
        Map<String, Object> out = FillToolSchemaBuilder
            .buildInputSchema(schema("lex_query", spec("type", "string", "required", true, "description", "content words only")));

        assertEquals("object", out.get("type"));
        assertEquals("string", property(out, "lex_query").get("type"));
        assertEquals("content words only", property(out, "lex_query").get("description"));
    }

    /** Only required params are listed; an optional is simply absent so the fill can omit it. */
    @Test
    public void build_listsOnlyRequiredParams() {
        Map<String, Object> out = FillToolSchemaBuilder
            .buildInputSchema(
                schema(
                    "lex_query",
                    spec("type", "string", "required", true),
                    "size",
                    spec("type", "number"),
                    "color",
                    spec("type", "string")
                )
            );

        assertEquals(List.of("lex_query"), required(out));
        assertTrue(properties(out).containsKey("size"));
    }

    /** Property names are the real Mustache names, since those are the keys the render needs. */
    @Test
    public void build_keepsRealParamNamesIncludingDots() {
        Map<String, Object> out = FillToolSchemaBuilder.buildInputSchema(schema("author.first_name", spec("type", "string")));

        assertTrue(properties(out).containsKey("author.first_name"));
    }

    @Test
    public void build_mapsDeclaredTypesToJsonTypes() {
        Map<String, Object> out = FillToolSchemaBuilder
            .buildInputSchema(
                schema(
                    "s",
                    spec("type", "string"),
                    "n",
                    spec("type", "number"),
                    "i",
                    spec("type", "integer"),
                    "l",
                    spec("type", "long"),
                    "f",
                    spec("type", "float"),
                    "b",
                    spec("type", "boolean"),
                    "kw",
                    spec("type", "keyword")
                )
            );

        assertEquals("string", property(out, "s").get("type"));
        assertEquals("number", property(out, "n").get("type"));
        assertEquals("integer", property(out, "i").get("type"));
        assertEquals("integer", property(out, "l").get("type"));
        assertEquals("number", property(out, "f").get("type"));
        assertEquals("boolean", property(out, "b").get("type"));
        assertEquals("string", property(out, "kw").get("type"));
    }

    /** An array slot travels as a string holding a JSON array literal. */
    @Test
    public void build_declaresArrayParamsAsStringsWithAHint() {
        Map<String, Object> out = FillToolSchemaBuilder.buildInputSchema(schema("filters", spec("type", "array")));

        assertEquals("string", property(out, "filters").get("type"));
        assertTrue(String.valueOf(property(out, "filters").get("description")).contains("JSON array literal"));
    }

    @Test
    public void build_appendsTheArrayHintToAnExistingDescription() {
        Map<String, Object> out = FillToolSchemaBuilder
            .buildInputSchema(schema("filters", spec("type", "array", "description", "Extra filter clauses.")));

        String description = String.valueOf(property(out, "filters").get("description"));
        assertTrue(description.startsWith("Extra filter clauses."));
        assertTrue(description.contains("JSON array literal"));
    }

    @Test
    public void build_carriesEnumsThrough() {
        Map<String, Object> out = FillToolSchemaBuilder
            .buildInputSchema(schema("sort_order", spec("type", "string", "enum", List.of("asc", "desc"))));

        assertEquals(List.of("asc", "desc"), property(out, "sort_order").get("enum"));
        assertEquals("string", property(out, "sort_order").get("type"));
    }

    /** A numeric enum keeps its declared type, or the model would emit "5" and fail the value check. */
    @Test
    public void build_keepsTheDeclaredTypeForANumericEnum() {
        Map<String, Object> out = FillToolSchemaBuilder.buildInputSchema(schema("size", spec("type", "number", "enum", List.of(1, 5, 10))));

        assertEquals("number", property(out, "size").get("type"));
        assertEquals(List.of(1, 5, 10), property(out, "size").get("enum"));
    }

    /** Mixed members under a string type: drop the type and let the enum constrain the slot. */
    @Test
    public void build_dropsTheTypeForAMixedEnumUnderAStringType() {
        Map<String, Object> out = FillToolSchemaBuilder.buildInputSchema(schema("mixed", spec("type", "string", "enum", List.of("a", 1))));

        assertFalse(property(out, "mixed").containsKey("type"));
        assertEquals(List.of("a", 1), property(out, "mixed").get("enum"));
    }

    @Test
    public void build_ignoresAnEmptyEnum() {
        Map<String, Object> out = FillToolSchemaBuilder
            .buildInputSchema(schema("sort_order", spec("type", "string", "enum", new ArrayList<>())));

        assertFalse(property(out, "sort_order").containsKey("enum"));
        assertEquals("string", property(out, "sort_order").get("type"));
    }

    // ---- the abstain flag --------------------------------------------------

    @Test
    public void build_appendsTheAbstainFlagLast() {
        Map<String, Object> out = FillToolSchemaBuilder
            .buildInputSchema(schema("lex_query", spec("type", "string", "required", true), "size", spec("type", "number")));

        List<String> names = new ArrayList<>(properties(out).keySet());
        assertEquals(FillToolSchemaBuilder.CANNOT_EXPRESS_FIELD, names.get(names.size() - 1));
        assertEquals("boolean", property(out, FillToolSchemaBuilder.CANNOT_EXPRESS_FIELD).get("type"));
        assertEquals(false, property(out, FillToolSchemaBuilder.CANNOT_EXPRESS_FIELD).get("default"));
    }

    /** Abstaining must stay optional, or every fill would have to decide it explicitly. */
    @Test
    public void build_neverRequiresTheAbstainFlag() {
        Map<String, Object> out = FillToolSchemaBuilder.buildInputSchema(schema("lex_query", spec("type", "string", "required", true)));

        assertFalse(required(out).contains(FillToolSchemaBuilder.CANNOT_EXPRESS_FIELD));
    }

    /** The enumerated trigger list is load-bearing; a vaguer wording over-abstained in a sweep. */
    @Test
    public void abstainDescription_enumeratesItsTriggers() {
        String description = FillToolSchemaBuilder.CANNOT_EXPRESS_DESCRIPTION;
        assertTrue(description.contains("(a)"));
        assertTrue(description.contains("(b)"));
        assertTrue(description.contains("(c)"));
        assertTrue(description.contains("(d)"));
        assertTrue(description.contains("(e)"));
        assertTrue(description.contains("Otherwise leave it false"));
    }

    /** A real param of that name wins and the hatch is disabled, rather than one shadowing the other. */
    @Test
    public void build_whenTemplateOwnsTheAbstainName_disablesTheHatch() {
        Map<String, Object> paramSchema = schema(
            "lex_query",
            spec("type", "string", "required", true),
            FillToolSchemaBuilder.CANNOT_EXPRESS_FIELD,
            spec("type", "string", "description", "a real slot")
        );

        assertTrue(FillToolSchemaBuilder.abstainDisabled(paramSchema));

        Map<String, Object> out = FillToolSchemaBuilder.buildInputSchema(paramSchema);
        assertEquals("string", property(out, FillToolSchemaBuilder.CANNOT_EXPRESS_FIELD).get("type"));
        assertEquals("a real slot", property(out, FillToolSchemaBuilder.CANNOT_EXPRESS_FIELD).get("description"));
    }

    @Test
    public void abstainDisabled_isFalseForAnOrdinaryTemplate() {
        assertFalse(FillToolSchemaBuilder.abstainDisabled(schema("lex_query", spec("type", "string"))));
        assertFalse(FillToolSchemaBuilder.abstainDisabled(null));
    }

    // ---- degenerate schemas ------------------------------------------------

    @Test
    public void build_rejectsAnEmptySchema() {
        assertThrows(IllegalArgumentException.class, () -> FillToolSchemaBuilder.buildInputSchema(Map.of()));
        assertThrows(IllegalArgumentException.class, () -> FillToolSchemaBuilder.buildInputSchema(null));
    }

    @Test
    public void build_skipsMalformedEntriesButKeepsTheRest() {
        Map<String, Object> out = FillToolSchemaBuilder
            .buildInputSchema(schema("lex_query", spec("type", "string"), "broken", "not-an-object"));

        assertTrue(properties(out).containsKey("lex_query"));
        assertFalse(properties(out).containsKey("broken"));
    }

    @Test
    public void build_rejectsASchemaOfOnlyMalformedEntries() {
        IllegalArgumentException e = assertThrows(
            IllegalArgumentException.class,
            () -> FillToolSchemaBuilder.buildInputSchema(schema("broken", "not-an-object"))
        );
        assertTrue(e.getMessage().contains("no usable params"));
    }

    @Test
    public void build_omitsAnEmptyDescription() {
        Map<String, Object> out = FillToolSchemaBuilder.buildInputSchema(schema("lex_query", spec("type", "string", "description", "")));

        assertFalse(property(out, "lex_query").containsKey("description"));
    }
}
