/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.engine.tools.templatefill;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

import java.util.Map;

import org.junit.Before;
import org.junit.Test;
import org.mockito.ArgumentCaptor;
import org.mockito.Mock;
import org.mockito.MockitoAnnotations;
import org.opensearch.core.xcontent.NamedXContentRegistry;
import org.opensearch.script.Script;
import org.opensearch.script.ScriptService;
import org.opensearch.script.ScriptType;
import org.opensearch.script.TemplateScript;

public class StoredTemplateRendererTests {

    @Mock
    private ScriptService scriptService;

    private StoredTemplateRenderer renderer;

    @Before
    public void setup() {
        MockitoAnnotations.openMocks(this);
        renderer = new StoredTemplateRenderer(scriptService, NamedXContentRegistry.EMPTY);
    }

    @Test
    public void render_returnsParsedSearchBody() {
        stubRender("{\"size\":5,\"query\":{\"match\":{\"title\":\"tents\"}}}");

        Map<String, Object> body = renderer.render("tmpl", Map.of("lex_query", "tents", "size", 5));

        assertEquals(5, body.get("size"));
        assertTrue(body.get("query") instanceof Map);
    }

    /**
     * STORED rather than INLINE: resolving by id avoids a GetStoredScript round trip and
     * keeps the query path off the inline compilation rate limit.
     */
    @Test
    public void render_compilesTheStoredScriptById() {
        stubRender("{\"query\":{\"match_all\":{}}}");

        renderer.render("my_template", Map.of());

        ArgumentCaptor<Script> captor = ArgumentCaptor.forClass(Script.class);
        verify(scriptService).compile(captor.capture(), any());
        Script script = captor.getValue();
        assertEquals(ScriptType.STORED, script.getType());
        assertEquals("my_template", script.getIdOrCode());
        assertNull("a stored script must not carry a lang", script.getLang());
    }

    @Test
    public void render_passesFilledParamsToTheTemplate() {
        TemplateScript.Factory factory = mock(TemplateScript.Factory.class);
        TemplateScript templateScript = mock(TemplateScript.class);
        when(scriptService.compile(any(Script.class), any())).thenReturn(factory);
        when(factory.newInstance(any())).thenReturn(templateScript);
        when(templateScript.execute()).thenReturn("{\"query\":{\"match_all\":{}}}");

        Map<String, Object> params = Map.of("lex_query", "tents", "size", 5);
        renderer.render("tmpl", params);

        @SuppressWarnings("unchecked")
        ArgumentCaptor<Map<String, Object>> captor = ArgumentCaptor.forClass(Map.class);
        verify(factory).newInstance(captor.capture());
        assertEquals(params, captor.getValue());
    }

    @Test
    public void render_whenCompileFails_throwsWithTemplateId() {
        when(scriptService.compile(any(Script.class), any())).thenThrow(new IllegalStateException("no such stored script"));

        IllegalArgumentException e = assertThrows(IllegalArgumentException.class, () -> renderer.render("missing", Map.of()));
        assertTrue(e.getMessage().contains("missing"));
        assertTrue(e.getMessage().contains("failed to render"));
    }

    @Test
    public void render_whenExecuteFails_throwsWithTemplateId() {
        TemplateScript.Factory factory = mock(TemplateScript.Factory.class);
        TemplateScript templateScript = mock(TemplateScript.class);
        when(scriptService.compile(any(Script.class), any())).thenReturn(factory);
        when(factory.newInstance(any())).thenReturn(templateScript);
        when(templateScript.execute()).thenThrow(new IllegalArgumentException("bad section"));

        IllegalArgumentException e = assertThrows(IllegalArgumentException.class, () -> renderer.render("tmpl", Map.of()));
        assertTrue(e.getMessage().contains("failed to render"));
    }

    /** Fail closed: a fill that breaks the body's JSON must not reach the cluster as a query. */
    @Test
    public void render_whenRenderIsNotJson_throws() {
        stubRender("{\"size\":,}");

        IllegalArgumentException e = assertThrows(IllegalArgumentException.class, () -> renderer.render("tmpl", Map.of()));
        assertTrue(e.getMessage().contains("rendered invalid JSON"));
    }

    /**
     * A top-level array is legal JSON, and {@code parser.map()} yields an empty map for it
     * rather than failing, so the object check has to be explicit.
     */
    @Test
    public void render_whenRenderIsNotAnObject_throws() {
        stubRender("[1,2,3]");

        IllegalArgumentException e = assertThrows(IllegalArgumentException.class, () -> renderer.render("tmpl", Map.of()));
        assertTrue(e.getMessage().contains("did not render a JSON object"));
    }

    @Test
    public void render_whenRenderIsAScalar_throws() {
        stubRender("42");

        IllegalArgumentException e = assertThrows(IllegalArgumentException.class, () -> renderer.render("tmpl", Map.of()));
        assertTrue(e.getMessage().contains("did not render a JSON object"));
    }

    @Test
    public void render_whenRenderIsBlank_throws() {
        stubRender("   ");

        IllegalArgumentException e = assertThrows(IllegalArgumentException.class, () -> renderer.render("tmpl", Map.of()));
        assertTrue(e.getMessage().contains("rendered nothing"));
    }

    private void stubRender(String rendered) {
        TemplateScript.Factory factory = mock(TemplateScript.Factory.class);
        TemplateScript templateScript = mock(TemplateScript.class);
        when(scriptService.compile(any(Script.class), any())).thenReturn(factory);
        when(factory.newInstance(any())).thenReturn(templateScript);
        when(templateScript.execute()).thenReturn(rendered);
    }
}
