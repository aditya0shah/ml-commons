/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.engine.tools.templatefill;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

import java.util.concurrent.atomic.AtomicLong;

import org.junit.Before;
import org.junit.Test;
import org.mockito.ArgumentCaptor;
import org.mockito.Mock;
import org.mockito.MockitoAnnotations;
import org.mockito.stubbing.Answer;
import org.opensearch.action.get.GetRequest;
import org.opensearch.action.get.GetResponse;
import org.opensearch.common.settings.Settings;
import org.opensearch.common.util.concurrent.ThreadContext;
import org.opensearch.core.action.ActionListener;
import org.opensearch.core.common.bytes.BytesArray;
import org.opensearch.core.xcontent.NamedXContentRegistry;
import org.opensearch.index.IndexNotFoundException;
import org.opensearch.ml.common.agenticsearch.AgenticSearchTemplate;
import org.opensearch.threadpool.ThreadPool;
import org.opensearch.transport.client.Client;

public class TemplateSchemaResolverTests {

    private static final String DOC = "{\"template_id\":\"tmpl\",\"index_binding\":\"products\","
        + "\"description\":\"Full-text search over title.\","
        + "\"param_schema\":{\"lex_query\":{\"type\":\"string\",\"required\":true,\"description\":\"\"}}}";

    @Mock
    private Client client;
    @Mock
    private ThreadPool threadPool;

    private AtomicLong now;
    private TemplateSchemaResolver resolver;

    @Before
    public void setup() {
        MockitoAnnotations.openMocks(this);
        when(client.threadPool()).thenReturn(threadPool);
        when(threadPool.getThreadContext()).thenReturn(new ThreadContext(Settings.builder().build()));
        now = new AtomicLong(0L);
        resolver = new TemplateSchemaResolver(client, NamedXContentRegistry.EMPTY, 60_000L, now::get);
    }

    @Test
    public void resolve_readsTheSchemaDocById() {
        stubGet(DOC);

        AgenticSearchTemplate template = resolveOk("tmpl");

        assertEquals("tmpl", template.getTemplateId());
        assertEquals("products", template.getIndexBinding());
        assertEquals("Full-text search over title.", template.getDescription());
        assertTrue(template.getParamSchema().containsKey("lex_query"));

        ArgumentCaptor<GetRequest> captor = ArgumentCaptor.forClass(GetRequest.class);
        verify(client).get(captor.capture(), any());
        assertEquals(".plugins-ml-agentic-search-templates", captor.getValue().index());
        assertEquals("tmpl", captor.getValue().id());
    }

    /** A system index is unreadable under an ordinary user's context, so the read stashes. */
    @Test
    public void resolve_stashesTheThreadContext() {
        stubGet(DOC);

        resolveOk("tmpl");

        verify(client).threadPool();
        verify(threadPool).getThreadContext();
    }

    @Test
    public void resolve_cachesWithinTheTtl() {
        stubGet(DOC);

        resolveOk("tmpl");
        now.set(59_999L);
        resolveOk("tmpl");

        verify(client, times(1)).get(any(GetRequest.class), any());
    }

    @Test
    public void resolve_refetchesAfterTheTtl() {
        stubGet(DOC);

        resolveOk("tmpl");
        now.set(60_000L);
        resolveOk("tmpl");

        verify(client, times(2)).get(any(GetRequest.class), any());
    }

    @Test
    public void resolve_cachesPerTemplateId() {
        stubGet(DOC);

        resolveOk("tmpl");
        resolveOk("other");

        verify(client, times(2)).get(any(GetRequest.class), any());
    }

    @Test
    public void resolve_whenTemplateIsUnregistered_fails() {
        GetResponse response = mock(GetResponse.class);
        when(response.isExists()).thenReturn(false);
        stubGetResponse(response);

        Exception e = resolveFailure("tmpl");
        assertTrue(e instanceof IllegalArgumentException);
        assertTrue(e.getMessage().contains("no param-schema registered for template_id 'tmpl'"));
    }

    /** The index is created lazily at first register, so its absence means "nothing registered". */
    @Test
    public void resolve_whenIndexIsMissing_failsAsUnregistered() {
        stubGetFailure(new IndexNotFoundException(".plugins-ml-agentic-search-templates"));

        Exception e = resolveFailure("tmpl");
        assertTrue(e instanceof IllegalArgumentException);
        assertTrue(e.getMessage().contains("no param-schema registered"));
    }

    @Test
    public void resolve_whenParamSchemaIsEmpty_fails() {
        stubGet("{\"template_id\":\"tmpl\",\"param_schema\":{}}");

        Exception e = resolveFailure("tmpl");
        assertTrue(e.getMessage().contains("missing or empty"));
    }

    @Test
    public void resolve_whenParamSchemaIsAbsent_fails() {
        stubGet("{\"template_id\":\"tmpl\",\"description\":\"no slots\"}");

        Exception e = resolveFailure("tmpl");
        assertTrue(e.getMessage().contains("missing or empty"));
    }

    @Test
    public void resolve_whenDocIsUnparseable_fails() {
        stubGet("{\"template_id\":");

        Exception e = resolveFailure("tmpl");
        assertTrue(e.getMessage().contains("unreadable"));
    }

    @Test
    public void resolve_whenIdIsMissing_failsWithoutReading() {
        Exception e = resolveFailure(null);
        assertTrue(e.getMessage().contains("template_id is required"));
        verify(client, times(0)).get(any(GetRequest.class), any());
    }

    @Test
    public void resolve_whenIdIsBlank_failsWithoutReading() {
        Exception e = resolveFailure("  ");
        assertTrue(e.getMessage().contains("template_id is required"));
        verify(client, times(0)).get(any(GetRequest.class), any());
    }

    /** A failed read must not be cached, or one transient error would stick for a whole TTL. */
    @Test
    public void resolve_doesNotCacheFailures() {
        stubGetFailure(new IndexNotFoundException(".plugins-ml-agentic-search-templates"));
        resolveFailure("tmpl");

        stubGet(DOC);
        AgenticSearchTemplate template = resolveOk("tmpl");

        assertEquals("tmpl", template.getTemplateId());
        verify(client, times(2)).get(any(GetRequest.class), any());
    }

    @Test
    public void resolve_propagatesUnexpectedReadFailures() {
        stubGetFailure(new RuntimeException("circuit breaking"));

        Exception e = resolveFailure("tmpl");
        assertTrue(e instanceof RuntimeException);
        assertEquals("circuit breaking", e.getMessage());
    }

    // ---- helpers -----------------------------------------------------------

    private AgenticSearchTemplate resolveOk(String templateId) {
        Holder holder = new Holder();
        resolver.resolve(templateId, ActionListener.wrap(t -> holder.value = t, e -> holder.error = e));
        assertNull(holder.error == null ? null : holder.error.getMessage(), holder.error);
        assertNotNull(holder.value);
        return holder.value;
    }

    private Exception resolveFailure(String templateId) {
        Holder holder = new Holder();
        resolver.resolve(templateId, ActionListener.wrap(t -> holder.value = t, e -> holder.error = e));
        assertNotNull("expected a failure", holder.error);
        return holder.error;
    }

    private void stubGet(String source) {
        GetResponse response = mock(GetResponse.class);
        when(response.isExists()).thenReturn(true);
        when(response.isSourceEmpty()).thenReturn(false);
        when(response.getSourceAsBytesRef()).thenReturn(new BytesArray(source));
        stubGetResponse(response);
    }

    private void stubGetResponse(GetResponse response) {
        doAnswer((Answer<Void>) inv -> {
            ActionListener<GetResponse> listener = inv.getArgument(1);
            listener.onResponse(response);
            return null;
        }).when(client).get(any(GetRequest.class), any());
    }

    private void stubGetFailure(Exception failure) {
        doAnswer((Answer<Void>) inv -> {
            ActionListener<GetResponse> listener = inv.getArgument(1);
            listener.onFailure(failure);
            return null;
        }).when(client).get(any(GetRequest.class), any());
    }

    private static final class Holder {
        private AgenticSearchTemplate value;
        private Exception error;
    }
}
