/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.engine.tools.templatefill;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.LongSupplier;

import org.opensearch.action.get.GetRequest;
import org.opensearch.common.util.concurrent.ThreadContext;
import org.opensearch.common.xcontent.LoggingDeprecationHandler;
import org.opensearch.core.action.ActionListener;
import org.opensearch.core.common.bytes.BytesReference;
import org.opensearch.core.xcontent.MediaTypeRegistry;
import org.opensearch.core.xcontent.NamedXContentRegistry;
import org.opensearch.core.xcontent.XContentParser;
import org.opensearch.index.IndexNotFoundException;
import org.opensearch.ml.common.CommonValue;
import org.opensearch.ml.common.agenticsearch.AgenticSearchTemplate;
import org.opensearch.transport.client.Client;

import lombok.extern.log4j.Log4j2;

/**
 * Reads a registered template's param-schema at query time, with a short TTL cache.
 *
 * <p>The read is a get-by-id against the {@code .plugins-ml-agentic-search-templates}
 * system index — the doc id is the {@code template_id}, which is also the core
 * {@code _scripts} template name, so one identifier resolves both the schema and the
 * body. Registration already did the expensive work (parsing the body, fetching the
 * mapping, pre-flight rendering); the query path only reads the result.
 *
 * <p>The get runs with the thread context stashed, because a system index is not
 * readable under an ordinary user's context. The stash is deliberately scoped to just
 * this read: the caller goes on to read the user's own index mapping and sample
 * documents, and those must stay under the user's context. Note this means any user who
 * can run an agentic search can cause a template's schema to be read, since templates
 * carry no read-side ACL today.
 *
 * <p>Only successful resolutions are cached, keyed by template id alone. A schema edit
 * therefore takes at most one TTL to take effect, which matches the agent-server
 * implementation and keeps a hot template down to zero extra reads.
 */
@Log4j2
public class TemplateSchemaResolver {

    /** Matches the agent-server cache TTL. Long enough to amortize a hot template, short enough that an edit lands quickly. */
    public static final long DEFAULT_TTL_MILLIS = 60_000L;

    private static final String INDEX = CommonValue.ML_AGENTIC_SEARCH_TEMPLATES_INDEX;

    private final Client client;
    private final NamedXContentRegistry xContentRegistry;
    private final long ttlMillis;
    /** Monotonic clock in millis; injectable so tests can advance time without sleeping. */
    private final LongSupplier clock;
    private final Map<String, CacheEntry> cache = new ConcurrentHashMap<>();

    public TemplateSchemaResolver(Client client, NamedXContentRegistry xContentRegistry) {
        this(client, xContentRegistry, DEFAULT_TTL_MILLIS, () -> System.nanoTime() / 1_000_000L);
    }

    public TemplateSchemaResolver(Client client, NamedXContentRegistry xContentRegistry, long ttlMillis, LongSupplier clock) {
        this.client = client;
        this.xContentRegistry = xContentRegistry;
        this.ttlMillis = ttlMillis;
        this.clock = clock;
    }

    /**
     * Resolve one template's schema, from cache when fresh.
     *
     * <p>Fails rather than returning a partial result when the template is unregistered
     * or carries no usable schema — there is nothing to fill in either case. Callers are
     * expected to treat any failure as a signal to fall back to free-DSL generation.
     */
    public void resolve(String templateId, ActionListener<AgenticSearchTemplate> listener) {
        if (templateId == null || templateId.isBlank()) {
            listener.onFailure(new IllegalArgumentException("template_id is required to resolve a param-schema"));
            return;
        }
        CacheEntry cached = cache.get(templateId);
        if (cached != null && clock.getAsLong() < cached.expiresAtMillis) {
            listener.onResponse(cached.template);
            return;
        }
        fetch(templateId, ActionListener.wrap(template -> {
            cache.put(templateId, new CacheEntry(template, clock.getAsLong() + ttlMillis));
            listener.onResponse(template);
        }, listener::onFailure));
    }

    private void fetch(String templateId, ActionListener<AgenticSearchTemplate> listener) {
        try (ThreadContext.StoredContext ctx = client.threadPool().getThreadContext().stashContext()) {
            ActionListener<AgenticSearchTemplate> wrapped = ActionListener.runBefore(listener, ctx::restore);
            client.get(new GetRequest(INDEX, templateId), ActionListener.wrap(response -> {
                if (!response.isExists() || response.isSourceEmpty()) {
                    wrapped.onFailure(new IllegalArgumentException(notRegistered(templateId)));
                    return;
                }
                AgenticSearchTemplate template;
                try {
                    template = parse(response.getSourceAsBytesRef());
                } catch (Exception e) {
                    wrapped.onFailure(new IllegalArgumentException("param-schema for template_id '" + templateId + "' is unreadable", e));
                    return;
                }
                if (!template.hasParams()) {
                    wrapped
                        .onFailure(new IllegalArgumentException("param-schema for template_id '" + templateId + "' is missing or empty"));
                    return;
                }
                wrapped.onResponse(template);
            }, e -> {
                // The index is created lazily on first register, so its absence just means
                // nothing has been registered yet — the same outcome as an unknown id.
                if (e instanceof IndexNotFoundException) {
                    wrapped.onFailure(new IllegalArgumentException(notRegistered(templateId)));
                } else {
                    wrapped.onFailure(e);
                }
            }));
        }
    }

    private static String notRegistered(String templateId) {
        return "no param-schema registered for template_id '" + templateId + "'";
    }

    private AgenticSearchTemplate parse(BytesReference source) throws Exception {
        try (
            XContentParser parser = MediaTypeRegistry.JSON
                .xContent()
                .createParser(xContentRegistry, LoggingDeprecationHandler.INSTANCE, source.streamInput())
        ) {
            return AgenticSearchTemplate.parse(parser);
        }
    }

    private static final class CacheEntry {
        private final AgenticSearchTemplate template;
        private final long expiresAtMillis;

        private CacheEntry(AgenticSearchTemplate template, long expiresAtMillis) {
            this.template = template;
            this.expiresAtMillis = expiresAtMillis;
        }
    }
}
