package deepseek

import (
	"testing"

	"github.com/QuantumNous/new-api/common"
	"github.com/QuantumNous/new-api/dto"
	relaycommon "github.com/QuantumNous/new-api/relay/common"
)

func TestDeepSeekResponsesCompatRequiresExplicitToggle(t *testing.T) {
	request := dto.OpenAIResponsesRequest{
		Model: "deepseek-chat",
		Input: mustMarshalJSON(t, "hello"),
	}
	adaptor := Adaptor{}

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
	if !info.ResponsesToChatMode {
		t.Fatal("expected ResponsesToChatMode to be enabled after conversion")
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
