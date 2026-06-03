package openai

import (
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/QuantumNous/new-api/common"
	"github.com/QuantumNous/new-api/dto"
	"github.com/QuantumNous/new-api/logger"
	relaycommon "github.com/QuantumNous/new-api/relay/common"
	"github.com/QuantumNous/new-api/relay/helper"
	"github.com/QuantumNous/new-api/service"
	"github.com/QuantumNous/new-api/types"

	"github.com/gin-gonic/gin"
)

type sessionEntry struct {
	messages  []dto.Message
	timestamp time.Time
}

var (
	responsesSessionCache  sync.Map
	sessionTTL             = 30 * time.Minute
	sessionCleanupInterval = 10 * time.Minute
)

func init() {
	go func() {
		ticker := time.NewTicker(sessionCleanupInterval)
		defer ticker.Stop()
		for range ticker.C {
			now := time.Now()
			responsesSessionCache.Range(func(key, value any) bool {
				entry, ok := value.(*sessionEntry)
				if ok && now.Sub(entry.timestamp) > sessionTTL {
					responsesSessionCache.Delete(key)
				}
				return true
			})
		}
	}()
}

func loadPreviousMessages(previousResponseID string) []dto.Message {
	if previousResponseID == "" {
		return nil
	}
	val, ok := responsesSessionCache.Load(previousResponseID)
	if !ok {
		return nil
	}
	entry, ok := val.(*sessionEntry)
	if !ok {
		return nil
	}
	if time.Since(entry.timestamp) > sessionTTL {
		responsesSessionCache.Delete(previousResponseID)
		return nil
	}
	return cloneMessages(entry.messages)
}

func saveSessionMessages(responseID string, messages []dto.Message) {
	if responseID == "" || len(messages) == 0 {
		return
	}
	responsesSessionCache.Store(responseID, &sessionEntry{
		messages:  cloneMessages(messages),
		timestamp: time.Now(),
	})
}

func updateSessionTimestamp(responseID string) {
	if responseID == "" {
		return
	}
	val, ok := responsesSessionCache.Load(responseID)
	if !ok {
		return
	}
	entry, ok := val.(*sessionEntry)
	if ok {
		entry.timestamp = time.Now()
	}
}

var responsesBuiltinTools = map[string]bool{
	"web_search_preview":   true,
	"file_search":          true,
	"code_interpreter":     true,
	"computer_use_preview": true,
	"web_search":           true,
}

func isResponsesBuiltinTool(toolType string) bool {
	return responsesBuiltinTools[toolType]
}

func filterResponsesTools(rawToolsJSON []byte) []dto.ToolCallRequest {
	if len(rawToolsJSON) == 0 {
		return nil
	}
	var tools []map[string]any
	if err := common.Unmarshal(rawToolsJSON, &tools); err != nil {
		return nil
	}
	var filtered []dto.ToolCallRequest
	for _, tool := range tools {
		toolType, _ := tool["type"].(string)
		if toolType != "function" || isResponsesBuiltinTool(toolType) {
			continue
		}
		var tc dto.ToolCallRequest
		tc.Type = toolType
		if fn, ok := tool["function"].(map[string]any); ok {
			tc.Function.Name, _ = fn["name"].(string)
			tc.Function.Description, _ = fn["description"].(string)
			tc.Function.Parameters = fn["parameters"]
		}
		filtered = append(filtered, tc)
	}
	return filtered
}

func generateResponsesID() string {
	return "resp_" + common.GetRandomString(28)
}

func generateMessageID() string {
	return "msg_" + common.GetRandomString(24)
}

func ConvertResponsesToChatRequest(c *gin.Context, info *relaycommon.RelayInfo, request dto.OpenAIResponsesRequest) (*dto.GeneralOpenAIRequest, error) {
	chatReq := &dto.GeneralOpenAIRequest{
		Model: request.Model,
	}

	if len(request.Instructions) > 0 {
		var instructionsStr string
		if err := common.Unmarshal(request.Instructions, &instructionsStr); err == nil && instructionsStr != "" {
			chatReq.Messages = append(chatReq.Messages, dto.Message{
				Role:    "system",
				Content: instructionsStr,
			})
		} else {
			var instructionsObj map[string]any
			if err := common.Unmarshal(request.Instructions, &instructionsObj); err == nil {
				if text, ok := instructionsObj["text"].(string); ok && text != "" {
					chatReq.Messages = append(chatReq.Messages, dto.Message{
						Role:    "system",
						Content: text,
					})
				}
			}
		}
	}

	if request.PreviousResponseID != "" {
		if prevMessages := loadPreviousMessages(request.PreviousResponseID); len(prevMessages) > 0 {
			chatReq.Messages = append(chatReq.Messages, prevMessages...)
			updateSessionTimestamp(request.PreviousResponseID)
		}
	}

	if len(request.Input) > 0 {
		inputMessages := parseResponsesInput(request.Input)
		chatReq.Messages = append(chatReq.Messages, inputMessages...)
	}

	if len(chatReq.Messages) == 0 {
		chatReq.Messages = append(chatReq.Messages, dto.Message{
			Role:    "user",
			Content: "...",
		})
	}

	chatReq.Stream = request.Stream
	chatReq.Temperature = request.Temperature
	chatReq.TopP = request.TopP
	chatReq.StreamOptions = request.StreamOptions

	if request.MaxOutputTokens != nil {
		chatReq.MaxTokens = request.MaxOutputTokens
	}

	if len(request.Tools) > 0 {
		filteredTools := filterResponsesTools(request.Tools)
		if len(filteredTools) > 0 {
			chatReq.Tools = filteredTools
		}
	}

	if request.Reasoning != nil {
		chatReq.ReasoningEffort = request.Reasoning.Effort
		if info != nil {
			info.ReasoningEffort = request.Reasoning.Effort
		}
	}

	if info != nil {
		info.ResponsesToChatMode = true
		info.RequestURLPath = "/v1/chat/completions"
		info.ResponsesChatMessages = chatReq.Messages
	}

	return chatReq, nil
}

func parseResponsesInput(inputJSON []byte) []dto.Message {
	if len(inputJSON) == 0 {
		return nil
	}
	var messages []dto.Message

	if common.GetJsonType(inputJSON) == "string" {
		var str string
		if err := common.Unmarshal(inputJSON, &str); err == nil && str != "" {
			messages = append(messages, dto.Message{
				Role:    "user",
				Content: str,
			})
		}
		return messages
	}

	if common.GetJsonType(inputJSON) != "array" {
		return messages
	}

	var inputs []map[string]any
	if err := common.Unmarshal(inputJSON, &inputs); err != nil {
		return messages
	}

	for _, item := range inputs {
		itemType, _ := item["type"].(string)
		switch itemType {
		case "message":
			role, _ := item["role"].(string)
			if role == "" {
				role = "user"
			}
			content := parseMessageContent(item["content"])
			if content != nil {
				messages = append(messages, dto.Message{
					Role:    role,
					Content: content,
				})
			}
		case "input_text":
			text, _ := item["text"].(string)
			if text != "" {
				messages = append(messages, dto.Message{
					Role:    "user",
					Content: text,
				})
			}
		case "input_image":
			common.SysLog("Responses input_image not supported, skipping")
		case "input_file":
			common.SysLog("Responses input_file not supported, skipping")
		default:
			if _, hasRole := item["role"].(string); hasRole {
				role, _ := item["role"].(string)
				if role == "" {
					role = "user"
				}
				content := parseMessageContent(item["content"])
				if content != nil {
					messages = append(messages, dto.Message{
						Role:    role,
						Content: content,
					})
				}
			} else if text, ok := item["text"].(string); ok && text != "" {
				messages = append(messages, dto.Message{
					Role:    "user",
					Content: text,
				})
			}
		}
	}
	return messages
}

func parseMessageContent(content any) any {
	if content == nil {
		return nil
	}
	switch v := content.(type) {
	case string:
		return v
	case []any:
		var parts []any
		for _, item := range v {
			itemMap, ok := item.(map[string]any)
			if !ok {
				continue
			}
			contentType, _ := itemMap["type"].(string)
			if contentType == "input_text" || contentType == "output_text" || contentType == dto.ContentTypeText {
				text, _ := itemMap["text"].(string)
				parts = append(parts, map[string]any{
					"type": dto.ContentTypeText,
					"text": text,
				})
			}
		}
		if len(parts) > 0 {
			return parts
		}
		return nil
	default:
		return nil
	}
}

func cloneMessages(messages []dto.Message) []dto.Message {
	if len(messages) == 0 {
		return nil
	}
	cloned := make([]dto.Message, 0, len(messages))
	for _, msg := range messages {
		clonedMsg := msg
		switch content := msg.Content.(type) {
		case []any:
			parts := make([]any, len(content))
			copy(parts, content)
			clonedMsg.Content = parts
		}
		cloned = append(cloned, clonedMsg)
	}
	return cloned
}

func ChatToResponsesHandler(c *gin.Context, info *relaycommon.RelayInfo, resp *http.Response) (*dto.Usage, *types.NewAPIError) {
	defer service.CloseResponseBodyGracefully(resp)

	responseBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, types.NewOpenAIError(err, types.ErrorCodeReadResponseBodyFailed, http.StatusInternalServerError)
	}

	var chatResp dto.OpenAITextResponse
	if err := common.Unmarshal(responseBody, &chatResp); err != nil {
		return nil, types.NewOpenAIError(err, types.ErrorCodeBadResponseBody, http.StatusInternalServerError)
	}

	if oaiError := chatResp.GetOpenAIError(); oaiError != nil && oaiError.Type != "" {
		service.IOCopyBytesGracefully(c, resp, responseBody)
		return nil, types.WithOpenAIError(*oaiError, resp.StatusCode)
	}

	responsesID := generateResponsesID()
	msgID := generateMessageID()
	createdAt := time.Now().Unix()

	var outputText string
	var finishReason string
	if len(chatResp.Choices) > 0 {
		outputText = chatResp.Choices[0].Message.StringContent()
		finishReason = chatResp.Choices[0].FinishReason
		reasoningContent := chatResp.Choices[0].Message.GetReasoningContent()
		if reasoningContent != "" && outputText == "" {
			outputText = reasoningContent
		}
	}

	output := []dto.ResponsesOutput{{
		Type:   "message",
		ID:     msgID,
		Status: "completed",
		Role:   "assistant",
		Content: []dto.ResponsesOutputContent{{
			Type:        "output_text",
			Text:        outputText,
			Annotations: []interface{}{},
		}},
	}}

	usage := &dto.Usage{}
	if chatResp.Usage.PromptTokens > 0 || chatResp.Usage.CompletionTokens > 0 {
		usage.PromptTokens = chatResp.Usage.PromptTokens
		usage.CompletionTokens = chatResp.Usage.CompletionTokens
		usage.TotalTokens = chatResp.Usage.TotalTokens
		if usage.TotalTokens == 0 {
			usage.TotalTokens = usage.PromptTokens + usage.CompletionTokens
		}
		usage.InputTokens = chatResp.Usage.PromptTokens
		usage.OutputTokens = chatResp.Usage.CompletionTokens
		if chatResp.Usage.PromptTokensDetails.CachedTokens > 0 {
			usage.InputTokensDetails = &dto.InputTokenDetails{
				CachedTokens: chatResp.Usage.PromptTokensDetails.CachedTokens,
			}
		}
		if chatResp.Usage.CompletionTokenDetails.ReasoningTokens > 0 {
			usage.CompletionTokenDetails.ReasoningTokens = chatResp.Usage.CompletionTokenDetails.ReasoningTokens
		}
	} else {
		usage = service.ResponseText2Usage(c, outputText, info.UpstreamModelName, info.GetEstimatePromptTokens())
		usage.InputTokens = usage.PromptTokens
		usage.OutputTokens = usage.CompletionTokens
	}

	modelName := chatResp.Model
	if modelName == "" {
		modelName = info.UpstreamModelName
	}

	responsesResp := dto.OpenAIResponsesResponse{
		ID:        responsesID,
		Object:    "response",
		CreatedAt: int(createdAt),
		Status:    jsonRawMessageFromString("completed"),
		Model:     modelName,
		Output:    output,
		Usage:     usage,
	}

	if finishReason == "length" || finishReason == "max_tokens" {
		responsesResp.IncompleteDetails = &dto.IncompleteDetails{
			Reasoning: "max_output_tokens",
		}
	}

	respBody, err := common.Marshal(responsesResp)
	if err != nil {
		return nil, types.NewOpenAIError(err, types.ErrorCodeJsonMarshalFailed, http.StatusInternalServerError)
	}
	service.IOCopyBytesGracefully(c, resp, respBody)

	sessionMessages := buildSessionMessages(messagesFromInfo(info), outputText, responsesID)
	saveSessionMessages(responsesID, sessionMessages)

	return usage, nil
}

func ChatToResponsesStreamHandler(c *gin.Context, info *relaycommon.RelayInfo, resp *http.Response) (*dto.Usage, *types.NewAPIError) {
	if resp == nil || resp.Body == nil {
		return nil, types.NewOpenAIError(fmt.Errorf("invalid response"), types.ErrorCodeBadResponse, http.StatusInternalServerError)
	}
	defer service.CloseResponseBodyGracefully(resp)

	responsesID := generateResponsesID()
	msgID := generateMessageID()
	model := info.UpstreamModelName
	var createAt int64 = time.Now().Unix()

	var (
		usage              = &dto.Usage{}
		outputText         strings.Builder
		reasoningText      strings.Builder
		sentItemAdded      bool
		sentItemDone       bool
		sentCompleted      bool
		finishReason       string
		streamErr          *types.NewAPIError
		containStreamUsage bool
	)

	helper.StreamScannerHandler(c, resp, info, func(data string, sr *helper.StreamResult) {
		if streamErr != nil {
			sr.Stop(streamErr)
			return
		}

		var streamResp dto.ChatCompletionsStreamResponse
		if err := common.UnmarshalJsonStr(data, &streamResp); err != nil {
			logger.LogError(c, "failed to unmarshal chat stream response: "+err.Error())
			sr.Error(err)
			return
		}

		if streamResp.Model != "" {
			model = streamResp.Model
		}
		if streamResp.Created != 0 {
			createAt = streamResp.Created
		}

		if streamResp.Usage != nil && service.ValidUsage(streamResp.Usage) {
			containStreamUsage = true
			usage = streamResp.Usage
			usage.InputTokens = usage.PromptTokens
			usage.OutputTokens = usage.CompletionTokens
			if usage.TotalTokens == 0 {
				usage.TotalTokens = usage.PromptTokens + usage.CompletionTokens
			}
		}

		if len(streamResp.Choices) == 0 {
			return
		}

		choice := streamResp.Choices[0]

		if choice.FinishReason != nil && *choice.FinishReason != "" {
			finishReason = *choice.FinishReason
		}

		content := choice.Delta.GetContentString()
		reasoning := choice.Delta.GetReasoningContent()

		if content != "" {
			outputText.WriteString(content)
			if !sentItemAdded {
				sendResponsesCompatStreamData(c, dto.ResponsesStreamResponse{
					Type: "response.output_item.added",
					Item: &dto.ResponsesOutput{
						Type:   "message",
						ID:     msgID,
						Status: "in_progress",
						Role:   "assistant",
						Content: []dto.ResponsesOutputContent{{
							Type: "output_text", Text: "", Annotations: []interface{}{},
						}},
					},
				}, "")
				sentItemAdded = true
			}

			sendResponsesCompatStreamData(c, dto.ResponsesStreamResponse{
				Type:         "response.output_text.delta",
				Delta:        content,
				ItemID:       msgID,
				OutputIndex:  common.GetPointer(0),
				ContentIndex: common.GetPointer(0),
			}, "")
		}

		if reasoning != "" {
			reasoningText.WriteString(reasoning)
			if !sentItemAdded {
				sendResponsesCompatStreamData(c, dto.ResponsesStreamResponse{
					Type: "response.output_item.added",
					Item: &dto.ResponsesOutput{
						Type:   "message",
						ID:     msgID,
						Status: "in_progress",
						Role:   "assistant",
						Content: []dto.ResponsesOutputContent{{
							Type: "output_text", Text: "", Annotations: []interface{}{},
						}},
					},
				}, "")
				sentItemAdded = true
			}

			sendResponsesCompatStreamData(c, dto.ResponsesStreamResponse{
				Type:         "response.reasoning.delta",
				Delta:        reasoning,
				ItemID:       msgID,
				OutputIndex:  common.GetPointer(0),
				ContentIndex: common.GetPointer(0),
			}, "")
		}
	})

	if !sentItemAdded && outputText.Len() == 0 && reasoningText.Len() == 0 {
		sendResponsesCompatStreamData(c, dto.ResponsesStreamResponse{
			Type: "response.output_item.added",
			Item: &dto.ResponsesOutput{
				Type:   "message",
				ID:     msgID,
				Status: "completed",
				Role:   "assistant",
				Content: []dto.ResponsesOutputContent{{
					Type: "output_text", Text: "", Annotations: []interface{}{},
				}},
			},
		}, "")
	}

	if !sentItemDone {
		fullText := outputText.String()
		if reasoningText.Len() > 0 && fullText == "" {
			fullText = reasoningText.String()
		}
		sendResponsesCompatStreamData(c, dto.ResponsesStreamResponse{
			Type: "response.output_item.done",
			Item: &dto.ResponsesOutput{
				Type:   "message",
				ID:     msgID,
				Status: "completed",
				Role:   "assistant",
				Content: []dto.ResponsesOutputContent{{
					Type:        "output_text",
					Text:        fullText,
					Annotations: []interface{}{},
				}},
			},
		}, "")
		sentItemDone = true
	}

	completedText := outputText.String()
	if reasoningText.Len() > 0 && completedText == "" {
		completedText = reasoningText.String()
	}

	if !sentCompleted {
		if !containStreamUsage {
			text := outputText.String()
			if reasoningText.Len() > 0 {
				text += reasoningText.String()
			}
			usage = service.ResponseText2Usage(c, text, info.UpstreamModelName, info.GetEstimatePromptTokens())
			usage.InputTokens = usage.PromptTokens
			usage.OutputTokens = usage.CompletionTokens
		}

		completedResp := dto.OpenAIResponsesResponse{
			ID:        responsesID,
			Object:    "response",
			CreatedAt: int(createAt),
			Status:    jsonRawMessageFromString("completed"),
			Model:     model,
			Output: []dto.ResponsesOutput{{
				Type:   "message",
				ID:     msgID,
				Status: "completed",
				Role:   "assistant",
				Content: []dto.ResponsesOutputContent{{
					Type:        "output_text",
					Text:        completedText,
					Annotations: []interface{}{},
				}},
			}},
			Usage: usage,
		}

		if finishReason == "length" || finishReason == "max_tokens" {
			completedResp.IncompleteDetails = &dto.IncompleteDetails{
				Reasoning: "max_output_tokens",
			}
		}

		sendResponsesCompatStreamData(c, dto.ResponsesStreamResponse{
			Type:     "response.completed",
			Response: &completedResp,
		}, "")
		sentCompleted = true
	}

	helper.Done(c)

	sessionMessages := buildSessionMessages(messagesFromInfo(info), completedText, responsesID)
	saveSessionMessages(responsesID, sessionMessages)

	return usage, nil
}

func sendResponsesCompatStreamData(c *gin.Context, event dto.ResponsesStreamResponse, _ string) {
	helper.SetEventStreamHeaders(c)
	if event.Type != "" {
		c.Render(-1, common.CustomEvent{Data: fmt.Sprintf("event: %s\n", event.Type)})
	}
	jsonData, err := common.Marshal(event)
	if err != nil {
		logger.LogError(c, "failed to marshal responses stream event: "+err.Error())
		return
	}
	c.Render(-1, common.CustomEvent{Data: "data: " + string(jsonData)})
	_ = helper.FlushWriter(c)
}

func jsonRawMessageFromString(s string) []byte {
	data, _ := common.Marshal(s)
	return data
}

func buildSessionMessages(requestMessages []dto.Message, responseText, _ string) []dto.Message {
	messages := cloneMessages(requestMessages)
	messages = append(messages, dto.Message{
		Role:    "assistant",
		Content: responseText,
	})
	return messages
}

func messagesFromInfo(info *relaycommon.RelayInfo) []dto.Message {
	if info == nil {
		return nil
	}
	if len(info.ResponsesChatMessages) > 0 {
		return cloneMessages(info.ResponsesChatMessages)
	}
	if info.Request == nil {
		return nil
	}
	if req, ok := info.Request.(*dto.OpenAIResponsesRequest); ok {
		return parseResponsesInput(req.Input)
	}
	return nil
}
