/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.engine.tools;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

import java.util.HashMap;
import java.util.Map;

import org.junit.Before;
import org.junit.Test;
import org.mockito.ArgumentCaptor;
import org.mockito.Mock;
import org.mockito.MockitoAnnotations;
import org.mockito.stubbing.Answer;
import org.opensearch.action.admin.indices.get.GetIndexResponse;
import org.opensearch.action.search.SearchResponse;
import org.opensearch.cluster.metadata.MappingMetadata;
import org.opensearch.common.xcontent.XContentHelper;
import org.opensearch.common.xcontent.json.JsonXContent;
import org.opensearch.core.action.ActionListener;
import org.opensearch.ml.common.settings.MLFeatureEnabledSetting;
import org.opensearch.ml.engine.tools.templatefill.TemplateFillPlanner;
import org.opensearch.search.SearchHits;
import org.opensearch.transport.client.AdminClient;
import org.opensearch.transport.client.Client;
import org.opensearch.transport.client.IndicesAdminClient;

/**
 * The template_fill generation type: that it dispatches, that it takes its template from the
 * request ahead of registration, and above all that every way it can go wrong ends in query
 * planning rather than an error.
 */
public class QueryPlanningToolTemplateFillTests {

    private static final Map<String, Object> RENDERED = Map.of("size", 5L, "query", Map.of("match", Map.of("title", "tents")));

    @Mock
    private Client client;
    @Mock
    private MLModelTool queryGenerationTool;
    @Mock
    private TemplateFillPlanner planner;
    @Mock
    private MLFeatureEnabledSetting mlFeatureEnabledSetting;
    @Mock
    private AdminClient adminClient;
    @Mock
    private IndicesAdminClient indicesAdminClient;

    @Before
    public void setup() {
        MockitoAnnotations.openMocks(this);
        when(mlFeatureEnabledSetting.isAgenticSearchTemplateEnabled()).thenReturn(true);
        when(client.admin()).thenReturn(adminClient);
        when(adminClient.indices()).thenReturn(indicesAdminClient);
    }

    private QueryPlanningTool tool(String registeredTemplateId) {
        return new QueryPlanningTool(
            QueryPlanningTool.TEMPLATE_FILL_TYPE_FIELD,
            queryGenerationTool,
            client,
            null,
            null,
            registeredTemplateId,
            planner,
            mlFeatureEnabledSetting
        );
    }

    private static Map<String, String> params() {
        Map<String, String> parameters = new HashMap<>();
        parameters.put(QueryPlanningTool.QUESTION_FIELD, "cheap tents");
        parameters.put(QueryPlanningTool.INDEX_NAME_FIELD, "products");
        parameters.put(QueryPlanningTool.MODEL_ID_FIELD, "model-1");
        return parameters;
    }

    @Test
    public void run_returnsTheRenderedBody() {
        stubFillSucceeds();

        Object result = run(tool("product_search"), params());

        assertEquals(RENDERED, result);
        verify(queryGenerationTool, never()).run(any(), any());
    }

    @Test
    public void run_usesTheRegisteredTemplate() {
        stubFillSucceeds();

        run(tool("product_search"), params());

        ArgumentCaptor<String> templateId = ArgumentCaptor.forClass(String.class);
        verify(planner).fill(templateId.capture(), any(), any(), any());
        assertEquals("product_search", templateId.getValue());
    }

    /** A per-request template wins, which is the hook a per-request field would feed. */
    @Test
    public void run_prefersThePerRequestTemplate() {
        stubFillSucceeds();
        Map<String, String> parameters = params();
        parameters.put(QueryPlanningTool.TEMPLATE_ID_FIELD, "listing_search");

        run(tool("product_search"), parameters);

        ArgumentCaptor<String> templateId = ArgumentCaptor.forClass(String.class);
        verify(planner).fill(templateId.capture(), any(), any(), any());
        assertEquals("listing_search", templateId.getValue());
    }

    @Test
    public void run_passesTheModelIdThrough() {
        stubFillSucceeds();

        run(tool("product_search"), params());

        ArgumentCaptor<String> modelId = ArgumentCaptor.forClass(String.class);
        verify(planner).fill(any(), modelId.capture(), any(), any());
        assertEquals("model-1", modelId.getValue());
    }

    // ---- every failure lands in query planning ------------------------------

    @Test
    public void run_whenFillFails_generatesTheQueryInstead() {
        stubFillFails(new IllegalArgumentException("no param-schema registered"));
        stubQueryGenerationSucceeds();

        Object result = run(tool("product_search"), params());

        assertEquals("{\"query\":{\"match_all\":{}}}", result);
        verify(queryGenerationTool, times(1)).run(any(), any());
    }

    @Test
    public void run_whenModelAbstains_generatesTheQueryInstead() {
        stubFillFails(new TemplateFillPlanner.TemplateCannotExpressException("product_search"));
        stubQueryGenerationSucceeds();

        Object result = run(tool("product_search"), params());

        assertNotNull(result);
        verify(queryGenerationTool, times(1)).run(any(), any());
    }

    @Test
    public void run_whenNoTemplateIsAvailable_generatesTheQueryInstead() {
        stubQueryGenerationSucceeds();

        Object result = run(tool(null), params());

        assertNotNull(result);
        verify(planner, never()).fill(any(), any(), any(), any());
        verify(queryGenerationTool, times(1)).run(any(), any());
    }

    @Test
    public void run_whenRegisteredTemplateIsBlank_generatesTheQueryInstead() {
        stubQueryGenerationSucceeds();

        run(tool("  "), params());

        verify(planner, never()).fill(any(), any(), any(), any());
        verify(queryGenerationTool, times(1)).run(any(), any());
    }

    /** The fill path is not wired at all when the node did not supply a ScriptService. */
    @Test
    public void run_whenFillIsNotWired_generatesTheQueryInstead() {
        stubQueryGenerationSucceeds();
        QueryPlanningTool unwired = new QueryPlanningTool(
            QueryPlanningTool.TEMPLATE_FILL_TYPE_FIELD,
            queryGenerationTool,
            client,
            null,
            null
        );

        run(unwired, params());

        verify(queryGenerationTool, times(1)).run(any(), any());
    }

    /** The kill switch costs the fill, not the query. */
    @Test
    public void run_whenFeatureIsDisabled_generatesTheQueryInstead() {
        when(mlFeatureEnabledSetting.isAgenticSearchTemplateEnabled()).thenReturn(false);
        stubQueryGenerationSucceeds();

        run(tool("product_search"), params());

        verify(planner, never()).fill(any(), any(), any(), any());
        verify(queryGenerationTool, times(1)).run(any(), any());
    }

    /** The fill path needs neither, so a missing question is still the tool's own validation error. */
    @Test
    public void run_stillValidatesRequiredParameters() {
        Map<String, String> parameters = new HashMap<>();
        parameters.put(QueryPlanningTool.INDEX_NAME_FIELD, "products");

        Result result = new Result();
        tool("product_search").run(parameters, ActionListener.wrap(r -> result.value = r, e -> result.error = e));

        assertNotNull(result.error);
        assertTrue(result.error.getMessage().contains("Validation error"));
        verify(planner, never()).fill(any(), any(), any(), any());
    }

    // ---- factory -----------------------------------------------------------

    @Test
    public void factory_acceptsTheNewGenerationType() {
        QueryPlanningTool.Factory factory = QueryPlanningTool.Factory.getInstance();
        factory.init(client);
        Map<String, Object> params = new HashMap<>();
        params.put(QueryPlanningTool.MODEL_ID_FIELD, "model-1");
        params.put(QueryPlanningTool.GENERATION_TYPE_FIELD, QueryPlanningTool.TEMPLATE_FILL_TYPE_FIELD);
        params.put(QueryPlanningTool.TEMPLATE_ID_FIELD, "product_search");

        QueryPlanningTool created = factory.create(params);

        assertEquals(QueryPlanningTool.TEMPLATE_FILL_TYPE_FIELD, created.getGenerationType());
        assertEquals("product_search", created.getTemplateId());
    }

    @Test
    public void factory_stillRejectsAnUnknownGenerationType() {
        QueryPlanningTool.Factory factory = QueryPlanningTool.Factory.getInstance();
        factory.init(client);
        Map<String, Object> params = new HashMap<>();
        params.put(QueryPlanningTool.MODEL_ID_FIELD, "model-1");
        params.put(QueryPlanningTool.GENERATION_TYPE_FIELD, "nonsense");

        IllegalArgumentException e = assertThrows(IllegalArgumentException.class, () -> factory.create(params));
        assertTrue(e.getMessage().contains("template_fill"));
    }

    @Test
    public void factory_templateIdIsOptionalAtRegistration() {
        QueryPlanningTool.Factory factory = QueryPlanningTool.Factory.getInstance();
        factory.init(client);
        Map<String, Object> params = new HashMap<>();
        params.put(QueryPlanningTool.MODEL_ID_FIELD, "model-1");
        params.put(QueryPlanningTool.GENERATION_TYPE_FIELD, QueryPlanningTool.TEMPLATE_FILL_TYPE_FIELD);

        assertNull(factory.create(params).getTemplateId());
    }

    // ---- helpers -----------------------------------------------------------

    private Object run(QueryPlanningTool tool, Map<String, String> parameters) {
        Result result = new Result();
        tool.run(parameters, ActionListener.wrap(r -> result.value = r, e -> result.error = e));
        assertNull(result.error == null ? null : result.error.getMessage(), result.error);
        return result.value;
    }

    private void stubFillSucceeds() {
        doAnswer((Answer<Void>) inv -> {
            ActionListener<Map<String, Object>> l = inv.getArgument(3);
            l.onResponse(RENDERED);
            return null;
        }).when(planner).fill(any(), any(), any(), any());
    }

    private void stubFillFails(Exception failure) {
        doAnswer((Answer<Void>) inv -> {
            ActionListener<Map<String, Object>> l = inv.getArgument(3);
            l.onFailure(failure);
            return null;
        }).when(planner).fill(any(), any(), any(), any());
    }

    /** The free-DSL arm: mapping fetch, then sample doc, then the model. */
    private void stubQueryGenerationSucceeds() {
        GetIndexResponse getIndexResponse = mock(GetIndexResponse.class);
        MappingMetadata mapping = new MappingMetadata(
            "products",
            XContentHelper.convertToMap(JsonXContent.jsonXContent, "{\"properties\":{\"title\":{\"type\":\"text\"}}}", true)
        );
        when(getIndexResponse.mappings()).thenReturn(Map.of("products", mapping));
        doAnswer((Answer<Void>) inv -> {
            ActionListener<GetIndexResponse> l = inv.getArgument(1);
            l.onResponse(getIndexResponse);
            return null;
        }).when(indicesAdminClient).getIndex(any(), any());

        SearchResponse searchResponse = mock(SearchResponse.class);
        when(searchResponse.getHits()).thenReturn(SearchHits.empty());
        doAnswer((Answer<Void>) inv -> {
            ActionListener<SearchResponse> l = inv.getArgument(1);
            l.onResponse(searchResponse);
            return null;
        }).when(client).search(any(), any());

        doAnswer((Answer<Void>) inv -> {
            ActionListener<Object> l = inv.getArgument(1);
            l.onResponse("{\"query\":{\"match_all\":{}}}");
            return null;
        }).when(queryGenerationTool).run(any(), any());
    }

    private static final class Result {
        private Object value;
        private Exception error;
    }
}
