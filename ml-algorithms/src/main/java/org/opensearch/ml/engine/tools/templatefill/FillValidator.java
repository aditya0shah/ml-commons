/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.engine.tools.templatefill;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

import org.opensearch.ml.common.agenticsearch.ParamSchema;
import org.opensearch.ml.common.utils.StringUtils;

/**
 * Checks what the model emitted against the template's param-schema and shapes it into
 * the map the renderer takes.
 *
 * <p>The tool schema constrains the fill but does not guarantee it, and on the
 * multi-template path it cannot: that schema marks every param optional and namespaces
 * the names, so required-ness and enum membership are only enforced here. Three rules
 * carry most of the weight:
 *
 * <ul>
 *   <li>An unknown key is dropped, not an error. One hallucinated property should not cost
 *       a whole fill when the render-parse guard still backstops the result.</li>
 *   <li>An unset optional is omitted rather than emitted as null, so the body's
 *       inverted-section defaults and optional clauses render as authored.</li>
 *   <li>An integral number stays integral. A JSON parser hands back {@code 5} as a double,
 *       and a double renders as {@code 5.0}, which OpenSearch rejects wherever an integer
 *       is expected — {@code size} being the obvious case.</li>
 * </ul>
 *
 * <p>Values are coerced where the intent is unambiguous (a quoted number for a numeric
 * slot, {@code "true"} for a boolean, a real JSON array for an array slot) rather than
 * failing the fill, because the alternative is falling back to free-DSL over a difference
 * of notation. A value that cannot be coerced does fail: rendering it would produce a
 * query that is wrong rather than merely differently written.
 */
public final class FillValidator {

    private FillValidator() {}

    /**
     * Validate and shape a fill.
     *
     * @param paramSchema the stored param-schema
     * @param emitted what the model returned, keyed by real param name
     * @return render-ready params, in schema order, unset optionals omitted
     * @throws IllegalArgumentException if a required param is missing or a value does not fit its param
     */
    public static Map<String, Object> validate(Map<String, Object> paramSchema, Map<String, Object> emitted) {
        if (paramSchema == null || paramSchema.isEmpty()) {
            throw new IllegalArgumentException("param-schema is empty; nothing to fill");
        }
        Map<String, Object> params = new LinkedHashMap<>();
        for (Map.Entry<String, Object> entry : paramSchema.entrySet()) {
            String name = entry.getKey();
            Map<String, Object> spec = ParamSchema.specOf(entry.getValue());
            if (spec == null) {
                continue;
            }
            Object value = emitted == null ? null : emitted.get(name);
            if (value == null) {
                if (ParamSchema.isRequired(spec)) {
                    throw new IllegalArgumentException("required param '" + name + "' was not filled");
                }
                continue;
            }
            Object coerced = coerce(name, value, ParamSchema.typeOf(spec));
            List<?> enumValues = ParamSchema.enumOf(spec);
            if (enumValues != null && !enumContains(enumValues, coerced)) {
                throw new IllegalArgumentException("param '" + name + "' value '" + coerced + "' is not one of " + enumValues);
            }
            params.put(name, coerced);
        }
        return params;
    }

    private static Object coerce(String name, Object value, String declaredType) {
        String type = declaredType.toLowerCase(Locale.ROOT);
        switch (type) {
            case "integer":
            case "int":
            case "long":
                return asIntegral(name, value, type);
            case ParamSchema.TYPE_NUMBER:
                // The deriver's type for an unquoted scalar slot. Keep an integral value
                // integral so it does not render as 5.0 in an integer position. Note the
                // branches must not be a ternary: mixing a Long with a double there
                // triggers binary numeric promotion and widens the Long straight back.
                Number number = asNumber(name, value, type);
                Long integral = toIntegral(number);
                if (integral != null) {
                    return integral;
                }
                return number.doubleValue();
            case "float":
            case "double":
                return asNumber(name, value, type).doubleValue();
            case ParamSchema.TYPE_BOOLEAN:
            case "bool":
                return asBoolean(name, value);
            case ParamSchema.TYPE_ARRAY:
                return asRawJson(value);
            default:
                // string, text, keyword, and anything unrecognized.
                return value instanceof String ? value : String.valueOf(value);
        }
    }

    private static Object asIntegral(String name, Object value, String type) {
        Long integral = toIntegral(asNumber(name, value, type));
        if (integral == null) {
            throw new IllegalArgumentException("param '" + name + "' expects an integer but got '" + value + "'");
        }
        return integral;
    }

    private static Number asNumber(String name, Object value, String type) {
        if (value instanceof Number) {
            return (Number) value;
        }
        if (value instanceof String) {
            // A model will sometimes quote a number; the intent is unambiguous.
            try {
                return Double.valueOf(((String) value).trim());
            } catch (NumberFormatException e) {
                throw new IllegalArgumentException("param '" + name + "' expects " + type + " but got '" + value + "'", e);
            }
        }
        throw new IllegalArgumentException("param '" + name + "' expects " + type + " but got '" + value + "'");
    }

    /** The value as a whole number, or null when it has a fractional part or does not fit. */
    private static Long toIntegral(Number number) {
        if (number instanceof Integer || number instanceof Long || number instanceof Short || number instanceof Byte) {
            return number.longValue();
        }
        double d = number.doubleValue();
        if (Double.isNaN(d) || Double.isInfinite(d) || d != Math.rint(d) || Math.abs(d) > Long.MAX_VALUE) {
            return null;
        }
        return (long) d;
    }

    private static Boolean asBoolean(String name, Object value) {
        if (value instanceof Boolean) {
            return (Boolean) value;
        }
        if (value instanceof String) {
            String s = ((String) value).trim();
            if ("true".equalsIgnoreCase(s)) {
                return Boolean.TRUE;
            }
            if ("false".equalsIgnoreCase(s)) {
                return Boolean.FALSE;
            }
        }
        throw new IllegalArgumentException("param '" + name + "' expects a boolean but got '" + value + "'");
    }

    /**
     * An array param is a triple-stache slot, so its value must be a string holding raw
     * JSON. A real list is serialized rather than rejected: handing the Mustache engine a
     * List would stringify it into Java map notation, and because the value also guards its
     * own section a multi-element list would make that section iterate.
     */
    private static String asRawJson(Object value) {
        if (value instanceof String) {
            return (String) value;
        }
        return StringUtils.gson.toJson(value);
    }

    private static boolean enumContains(List<?> enumValues, Object coerced) {
        for (Object member : enumValues) {
            if (member == null) {
                continue;
            }
            if (member.equals(coerced)) {
                return true;
            }
            // The stored enum and the coerced value can be different numeric boxes
            // (the doc parses 5 as an Integer, the fill coerces to a Long).
            if (member instanceof Number && coerced instanceof Number) {
                if (((Number) member).doubleValue() == ((Number) coerced).doubleValue()) {
                    return true;
                }
                continue;
            }
            if (String.valueOf(member).equals(String.valueOf(coerced))) {
                return true;
            }
        }
        return false;
    }
}
