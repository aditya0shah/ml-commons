/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.common.agenticsearch;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

import java.util.List;
import java.util.Map;

import org.junit.Test;

public class ParamSchemaTests {

    @Test
    public void testValueFitsTypeNumber() {
        assertTrue(ParamSchema.valueFitsType(5, ParamSchema.TYPE_NUMBER));
        assertTrue(ParamSchema.valueFitsType(5.5, ParamSchema.TYPE_NUMBER));
        assertTrue(ParamSchema.valueFitsType(5L, ParamSchema.TYPE_NUMBER));
        assertFalse(ParamSchema.valueFitsType("5", ParamSchema.TYPE_NUMBER));
        assertFalse(ParamSchema.valueFitsType(true, ParamSchema.TYPE_NUMBER));
    }

    @Test
    public void testValueFitsTypeBoolean() {
        assertTrue(ParamSchema.valueFitsType(true, ParamSchema.TYPE_BOOLEAN));
        assertFalse(ParamSchema.valueFitsType("true", ParamSchema.TYPE_BOOLEAN));
        assertFalse(ParamSchema.valueFitsType(1, ParamSchema.TYPE_BOOLEAN));
    }

    /**
     * An array param is a triple-stache raw-JSON slot, so only a String fits. A List
     * renders as Java map-toString and makes the guarding section iterate.
     */
    @Test
    public void testValueFitsTypeArrayIsStringOnly() {
        assertTrue(ParamSchema.valueFitsType("[]", ParamSchema.TYPE_ARRAY));
        assertTrue(ParamSchema.valueFitsType("[{\"term\":{\"t\":\"a\"}}]", ParamSchema.TYPE_ARRAY));
        assertFalse(ParamSchema.valueFitsType(List.of("a", "b"), ParamSchema.TYPE_ARRAY));
        assertFalse(ParamSchema.valueFitsType(List.of(), ParamSchema.TYPE_ARRAY));
    }

    @Test
    public void testValueFitsTypeStringAndUnknownTypes() {
        assertTrue(ParamSchema.valueFitsType("x", ParamSchema.TYPE_STRING));
        assertFalse(ParamSchema.valueFitsType(1, ParamSchema.TYPE_STRING));
        // An unrecognized type falls through to the string rule.
        assertTrue(ParamSchema.valueFitsType("x", "geo_point"));
        assertFalse(ParamSchema.valueFitsType(1, "geo_point"));
    }

    @Test
    public void testValueFitsTypeNullNeverFits() {
        assertFalse(ParamSchema.valueFitsType(null, ParamSchema.TYPE_STRING));
        assertFalse(ParamSchema.valueFitsType(null, ParamSchema.TYPE_NUMBER));
        assertFalse(ParamSchema.valueFitsType(null, ParamSchema.TYPE_ARRAY));
    }

    @Test
    public void testTypeOfDefaultsToString() {
        assertEquals(ParamSchema.TYPE_NUMBER, ParamSchema.typeOf(Map.of("type", "number")));
        assertEquals(ParamSchema.TYPE_STRING, ParamSchema.typeOf(Map.of()));
        assertEquals(ParamSchema.TYPE_STRING, ParamSchema.typeOf(Map.of("type", "")));
        assertEquals(ParamSchema.TYPE_STRING, ParamSchema.typeOf(Map.of("type", 7)));
    }

    /** A caller-registered schema may omit required; absent means false. */
    @Test
    public void testIsRequiredAbsentMeansFalse() {
        assertTrue(ParamSchema.isRequired(Map.of("required", true)));
        assertFalse(ParamSchema.isRequired(Map.of("required", false)));
        assertFalse(ParamSchema.isRequired(Map.of()));
        assertFalse(ParamSchema.isRequired(Map.of("required", "true")));
    }

    @Test
    public void testDescriptionOfDefaultsToEmpty() {
        assertEquals("how many hits", ParamSchema.descriptionOf(Map.of("description", "how many hits")));
        assertEquals("", ParamSchema.descriptionOf(Map.of()));
        assertEquals("", ParamSchema.descriptionOf(Map.of("description", 7)));
    }

    @Test
    public void testEnumOfTreatsEmptyAndNonListAsAbsent() {
        assertEquals(List.of("asc", "desc"), ParamSchema.enumOf(Map.of("enum", List.of("asc", "desc"))));
        assertNull(ParamSchema.enumOf(Map.of()));
        assertNull(ParamSchema.enumOf(Map.of("enum", List.of())));
        assertNull(ParamSchema.enumOf(Map.of("enum", "asc")));
    }

    @Test
    public void testSpecOfRejectsNonObjectEntry() {
        assertEquals(Map.of("type", "string"), ParamSchema.specOf(Map.of("type", "string")));
        assertNull(ParamSchema.specOf("string"));
        assertNull(ParamSchema.specOf(null));
        assertNull(ParamSchema.specOf(List.of("string")));
    }
}
