package openai

import (
	"io"
	"net/http"
	"net/http/httptest"
	"regexp"
	"strings"
	"testing"

	"github.com/QuantumNous/new-api/common"
	"github.com/QuantumNous/new-api/constant"
	"github.com/QuantumNous/new-api/dto"
	relaycommon "github.com/QuantumNous/new-api/relay/common"
	"github.com/gin-gonic/gin"
)

func TestConvertResponsesToChatRequest(t *testing.T) {
	gin.SetMode(gin.ReleaseMode)
	clearResponsesSessionCache()

	saveSessionMessages("resp_prev", []dto.Message{
		{Role: "system", Content: "previous system"},
		{Role: "user", Content: "remember BANANA"},
		{Role: "assistant", Content: "remembered"},
	})

	maxTokens := uint(123)
	stream := true
	temperature := 0.7
	topP := 0.8
	request := dto.OpenAIResponsesRequest{
		Model:              "gpt-5.5",
		Instructions:       mustMarshalJSON(t, "current system"),
		Input:              mustMarshalJSON(t, []map[string]any{{"type": "input_text", "text": "next turn"}}),
		MaxOutputTokens:    &maxTokens,
		PreviousResponseID: "resp_prev",
		Stream:             &stream,
		Temperature:        &temperature,
		TopP:               &topP,
		Tools: mustMarshalJSON(t, []map[string]any{
			{
				"type": "function",
				"function": map[string]any{
					"name":        "lookup_weather",
					"description": "Look up weather",
					"parameters":  map[string]any{"type": "object"},
				},
			},
			{"type": "web_search_preview"},
		}),
		Reasoning: &dto.Reasoning{Effort: "high"},
	}
	info := &relaycommon.RelayInfo{}

	chatReq, err := ConvertResponsesToChatRequest(nil, info, request)
	if err != nil {
		t.Fatalf("ConvertResponsesToChatRequest returned error: %v", err)
	}

	if got, want := len(chatReq.Messages), 5; got != want {
		t.Fatalf("unexpected message count: got %d want %d", got, want)
	}
	if got := chatReq.Messages[0].StringContent(); got != "current system" {
		t.Fatalf("instructions were not mapped to first system message: %q", got)
	}
	if got := chatReq.Messages[1].StringContent(); got != "previous system" {
		t.Fatalf("previous system message missing from restored context: %q", got)
	}
	if got := chatReq.Messages[4].StringContent(); got != "next turn" {
		t.Fatalf("latest input message not appended: %q", got)
	}
	if chatReq.MaxTokens == nil || *chatReq.MaxTokens != maxTokens {
		t.Fatalf("max_output_tokens was not mapped to max_tokens: %#v", chatReq.MaxTokens)
	}
	if chatReq.Stream == nil || *chatReq.Stream != stream {
		t.Fatalf("stream flag was not propagated: %#v", chatReq.Stream)
	}
	if chatReq.Temperature == nil || *chatReq.Temperature != temperature {
		t.Fatalf("temperature was not propagated: %#v", chatReq.Temperature)
	}
	if chatReq.TopP == nil || *chatReq.TopP != topP {
		t.Fatalf("top_p was not propagated: %#v", chatReq.TopP)
	}
	if len(chatReq.Tools) != 1 || chatReq.Tools[0].Function.Name != "lookup_weather" {
		t.Fatalf("unexpected filtered tools: %#v", chatReq.Tools)
	}
	if !info.ResponsesToChatMode {
		t.Fatal("ResponsesToChatMode should be enabled")
	}
	if info.RequestURLPath != "/v1/chat/completions" {
		t.Fatalf("unexpected request path: %q", info.RequestURLPath)
	}
	if info.ReasoningEffort != "high" {
		t.Fatalf("reasoning effort not propagated: %q", info.ReasoningEffort)
	}
}

func TestConvertResponsesToChatRequestSupportsTopLevelFunctionTools(t *testing.T) {
	gin.SetMode(gin.ReleaseMode)

	request := dto.OpenAIResponsesRequest{
		Model: "gpt-5.4",
		Input: mustMarshalJSON(t, "hello"),
		Tools: mustMarshalJSON(t, []map[string]any{
			{
				"type":        "function",
				"name":        "apply_patch",
				"description": "Apply a patch",
				"parameters": map[string]any{
					"type": "object",
				},
			},
			{
				"type":        "function",
				"name":        "",
				"description": "invalid empty tool name",
				"parameters": map[string]any{
					"type": "object",
				},
			},
		}),
	}

	chatReq, err := ConvertResponsesToChatRequest(nil, &relaycommon.RelayInfo{}, request)
	if err != nil {
		t.Fatalf("ConvertResponsesToChatRequest returned error: %v", err)
	}
	if len(chatReq.Tools) != 1 {
		t.Fatalf("unexpected tool count: %#v", chatReq.Tools)
	}
	if chatReq.Tools[0].Function.Name != "apply_patch" {
		t.Fatalf("unexpected tool name: %#v", chatReq.Tools[0])
	}
}

func TestOpenAIResponsesCompatRequiresExplicitToggle(t *testing.T) {
	gin.SetMode(gin.ReleaseMode)

	request := dto.OpenAIResponsesRequest{
		Model: "deepseek-chat",
		Input: mustMarshalJSON(t, "hello"),
	}
	adaptor := Adaptor{ChannelType: constant.ChannelTypeOpenAI}

	got, err := adaptor.ConvertOpenAIResponsesRequest(nil, &relaycommon.RelayInfo{}, request)
	if err != nil {
		t.Fatalf("ConvertOpenAIResponsesRequest returned error: %v", err)
	}
	if _, ok := got.(dto.OpenAIResponsesRequest); !ok {
		t.Fatalf("expected raw responses request without compat toggle, got %T", got)
	}

	enabled := true
	info := &relaycommon.RelayInfo{
		ChannelMeta: &relaycommon.ChannelMeta{},
	}
	info.ChannelOtherSettings.ResponsesCompatMode = &enabled
	got, err = adaptor.ConvertOpenAIResponsesRequest(nil, info, request)
	if err != nil {
		t.Fatalf("ConvertOpenAIResponsesRequest with toggle returned error: %v", err)
	}
	chatReq, ok := got.(*dto.GeneralOpenAIRequest)
	if !ok {
		t.Fatalf("expected chat completions request with compat toggle, got %T", got)
	}
	if len(chatReq.Messages) != 1 || chatReq.Messages[0].StringContent() != "hello" {
		t.Fatalf("unexpected converted messages: %#v", chatReq.Messages)
	}
}

func TestChatToResponsesHandlerCachesFullConversation(t *testing.T) {
	gin.SetMode(gin.ReleaseMode)
	clearResponsesSessionCache()

	recorder := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(recorder)
	c.Request = httptest.NewRequest(http.MethodPost, "/v1/responses", nil)

	info := &relaycommon.RelayInfo{
		ChannelMeta: &relaycommon.ChannelMeta{
			UpstreamModelName: "deepseek-chat",
		},
		ResponsesChatMessages: []dto.Message{
			{Role: "system", Content: "Be terse"},
			{Role: "user", Content: "Say hello"},
		},
	}
	resp := &http.Response{
		StatusCode: http.StatusOK,
		Header:     make(http.Header),
		Body: io.NopCloser(strings.NewReader(`{
			"id":"chatcmpl-1",
			"object":"chat.completion",
			"created":123,
			"model":"deepseek-chat",
			"choices":[{"index":0,"message":{"role":"assistant","content":"Hello"},"finish_reason":"stop"}],
			"usage":{
				"prompt_tokens":5,
				"completion_tokens":1,
				"total_tokens":6,
				"prompt_tokens_details":{"cached_tokens":0,"text_tokens":0,"audio_tokens":0,"image_tokens":0},
				"completion_tokens_details":{"text_tokens":0,"audio_tokens":0,"image_tokens":0,"reasoning_tokens":0}
			}
		}`)),
	}

	usage, err := ChatToResponsesHandler(c, info, resp)
	if err != nil {
		t.Fatalf("ChatToResponsesHandler returned error: %v", err)
	}
	if usage == nil || usage.InputTokens != 5 || usage.OutputTokens != 1 {
		t.Fatalf("unexpected usage: %#v", usage)
	}

	var response dto.OpenAIResponsesResponse
	if err := common.Unmarshal(recorder.Body.Bytes(), &response); err != nil {
		t.Fatalf("failed to unmarshal responses payload: %v", err)
	}
	if response.ID == "" {
		t.Fatal("responses id was not generated")
	}

	cached := loadPreviousMessages(response.ID)
	if got, want := len(cached), 3; got != want {
		t.Fatalf("unexpected cached message count: got %d want %d", got, want)
	}
	if cached[0].Role != "system" || cached[0].StringContent() != "Be terse" {
		t.Fatalf("system message not preserved in cache: %#v", cached[0])
	}
	if cached[2].Role != "assistant" || cached[2].StringContent() != "Hello" {
		t.Fatalf("assistant reply not cached correctly: %#v", cached[2])
	}
}

func TestChatToResponsesStreamHandlerEmitsReasoningAndDone(t *testing.T) {
	gin.SetMode(gin.ReleaseMode)
	clearResponsesSessionCache()
	originalStreamingTimeout := constant.StreamingTimeout
	constant.StreamingTimeout = 30
	t.Cleanup(func() {
		constant.StreamingTimeout = originalStreamingTimeout
	})

	recorder := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(recorder)
	c.Request = httptest.NewRequest(http.MethodPost, "/v1/responses", nil)

	info := &relaycommon.RelayInfo{
		DisablePing: true,
		ChannelMeta: &relaycommon.ChannelMeta{
			UpstreamModelName: "deepseek-reasoner",
		},
		ResponsesChatMessages: []dto.Message{
			{Role: "system", Content: "Think step by step"},
			{Role: "user", Content: "What is 17 * 19?"},
		},
	}
	body := strings.Join([]string{
		`data: {"id":"chatcmpl-1","object":"chat.completion.chunk","created":123,"model":"deepseek-reasoner","choices":[{"index":0,"delta":{"reasoning_content":"compute"},"finish_reason":null}]}`,
		"",
		`data: {"id":"chatcmpl-1","object":"chat.completion.chunk","created":123,"model":"deepseek-reasoner","choices":[{"index":0,"delta":{"content":"323"},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":2,"total_tokens":7,"prompt_tokens_details":{"cached_tokens":0,"text_tokens":0,"audio_tokens":0,"image_tokens":0},"completion_tokens_details":{"text_tokens":0,"audio_tokens":0,"image_tokens":0,"reasoning_tokens":1}}}`,
		"",
		"data: [DONE]",
		"",
	}, "\n")
	resp := &http.Response{
		StatusCode: http.StatusOK,
		Header:     make(http.Header),
		Body:       io.NopCloser(strings.NewReader(body)),
	}

	usage, err := ChatToResponsesStreamHandler(c, info, resp)
	if err != nil {
		t.Fatalf("ChatToResponsesStreamHandler returned error: %v", err)
	}
	if usage == nil || usage.InputTokens != 5 || usage.OutputTokens != 2 {
		t.Fatalf("unexpected usage: %#v", usage)
	}

	streamOutput := recorder.Body.String()
	for _, expected := range []string{
		"event: response.output_item.added",
		"event: response.reasoning.delta",
		"event: response.output_text.delta",
		"event: response.output_item.done",
		"event: response.completed",
		"data: [DONE]",
	} {
		if !strings.Contains(streamOutput, expected) {
			t.Fatalf("stream output missing %q:\n%s", expected, streamOutput)
		}
	}

	respID := regexp.MustCompile(`resp_[A-Za-z0-9]+`).FindString(streamOutput)
	if respID == "" {
		t.Fatalf("failed to find generated response id in stream output:\n%s", streamOutput)
	}
	cached := loadPreviousMessages(respID)
	if got, want := len(cached), 3; got != want {
		t.Fatalf("unexpected cached stream message count: got %d want %d", got, want)
	}
	if cached[0].Role != "system" || cached[2].StringContent() != "323" {
		t.Fatalf("cached stream conversation mismatch: %#v", cached)
	}
}

func mustMarshalJSON(t *testing.T, v any) []byte {
	t.Helper()
	data, err := common.Marshal(v)
	if err != nil {
		t.Fatalf("failed to marshal test json: %v", err)
	}
	return data
}

func clearResponsesSessionCache() {
	responsesSessionCache.Range(func(key, value any) bool {
		responsesSessionCache.Delete(key)
		return true
	})
}
