/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.engine.tools.templatefill;

import java.util.Collections;
import java.util.Map;

import org.opensearch.common.xcontent.LoggingDeprecationHandler;
import org.opensearch.core.xcontent.MediaTypeRegistry;
import org.opensearch.core.xcontent.NamedXContentRegistry;
import org.opensearch.core.xcontent.XContentParser;
import org.opensearch.script.Script;
import org.opensearch.script.ScriptService;
import org.opensearch.script.ScriptType;
import org.opensearch.script.TemplateScript;

/**
 * Renders a stored Mustache search template with filled params, in process.
 *
 * <p>This is the same engine and the same code path as {@code POST _render/template}:
 * that REST endpoint's transport action compiles the body against
 * {@link TemplateScript#CONTEXT} and executes it, exactly as below. It is also the same
 * render the register-time pre-flight performs, so a body that passed registration
 * renders here the same way. Rendering in process rather than over REST is not a
 * preference — {@code lang-mustache} is a separately loaded OpenSearch module that
 * ml-commons neither depends on nor extends, so its request classes are not on the
 * compile classpath and there is no self-call available.
 *
 * <p>The script is compiled as {@link ScriptType#STORED}, resolved from cluster state by
 * id. That avoids a {@code GetStoredScript} round trip, and it sidesteps the inline
 * compilation rate limit (75 per 5 minutes over a 100-entry cache keyed by source), which
 * a large or churning template set could otherwise trip on the query path.
 *
 * <p>Escaping is deliberately left to the renderer. JSON-escaping filled values is the
 * Mustache engine's job, and doing it here as well would double-escape and reintroduce
 * the injection and escaping bugs that delegating avoids.
 */
public class StoredTemplateRenderer {

    private final ScriptService scriptService;
    private final NamedXContentRegistry xContentRegistry;

    public StoredTemplateRenderer(ScriptService scriptService, NamedXContentRegistry xContentRegistry) {
        this.scriptService = scriptService;
        this.xContentRegistry = xContentRegistry;
    }

    /**
     * Render the stored template and return the parsed search body.
     *
     * <p>Parsing the render back is a fail-closed check: a template whose slots were
     * filled with values that break its JSON must not reach the cluster as a query. The
     * check is JSON legality only, matching the register-time pre-flight; a rendered body
     * that is legal JSON but not a legal search body is caught downstream when
     * neural-search reparses it as a {@code SearchSourceBuilder}.
     *
     * @param templateId the stored script id, which is also the registered template id
     * @param params filled params keyed by their real Mustache names, unset optionals omitted
     * @return the rendered body parsed as a map
     * @throws IllegalArgumentException if the template will not render or the render is not a JSON object
     */
    public Map<String, Object> render(String templateId, Map<String, Object> params) {
        String rendered;
        try {
            Script script = new Script(ScriptType.STORED, null, templateId, Collections.emptyMap());
            TemplateScript.Factory factory = scriptService.compile(script, TemplateScript.CONTEXT);
            rendered = factory.newInstance(params).execute();
        } catch (Exception e) {
            throw new IllegalArgumentException("Template '" + templateId + "' failed to render: " + e.getMessage(), e);
        }
        if (rendered == null || rendered.isBlank()) {
            throw new IllegalArgumentException("Template '" + templateId + "' rendered nothing");
        }
        Map<String, Object> body;
        try (
            XContentParser parser = MediaTypeRegistry.JSON
                .xContent()
                .createParser(xContentRegistry, LoggingDeprecationHandler.INSTANCE, rendered)
        ) {
            // The token has to be checked explicitly: map() does not reject a top-level
            // array, it just yields an empty map, which would sail on as a valid-looking
            // empty search body.
            body = parser.nextToken() == XContentParser.Token.START_OBJECT ? parser.map() : null;
        } catch (Exception e) {
            throw new IllegalArgumentException("Template '" + templateId + "' rendered invalid JSON: " + e.getMessage(), e);
        }
        if (body == null) {
            throw new IllegalArgumentException("Template '" + templateId + "' did not render a JSON object");
        }
        return body;
    }
}
