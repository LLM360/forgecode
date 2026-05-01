use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use async_trait::async_trait;
use chrono::Utc;
use forge_domain::{
    Conversation, EndPayload, EventData, EventHandle, Provider, RequestPayload, ResponsePayload,
    StartPayload, ToolcallEndPayload, Transformer,
};
use tokio::io::AsyncWriteExt;
use tracing::warn;
use url::Url;

use crate::dto::openai::ProviderPipeline;

struct TraceState {
    turn: usize,
    started_at: Option<Instant>,
}

#[derive(Clone)]
pub struct TraceLoggingHandler {
    trace_file: Option<PathBuf>,
    state: Arc<Mutex<TraceState>>,
}

impl TraceLoggingHandler {
    pub fn new() -> Self {
        let trace_file = std::env::var("FORGE_TRACE_FILE")
            .ok()
            .filter(|s| !s.is_empty())
            .map(PathBuf::from);
        Self {
            trace_file,
            state: Arc::new(Mutex::new(TraceState { turn: 0, started_at: None })),
        }
    }

    #[cfg(test)]
    fn with_path(path: Option<PathBuf>) -> Self {
        Self {
            trace_file: path,
            state: Arc::new(Mutex::new(TraceState { turn: 0, started_at: None })),
        }
    }

    pub fn is_enabled(&self) -> bool {
        self.trace_file.is_some()
    }

    async fn write_event(&self, value: &serde_json::Value) {
        let path = match &self.trace_file {
            Some(p) => p,
            None => return,
        };
        let mut line = match serde_json::to_string(value) {
            Ok(s) => s,
            Err(e) => {
                warn!("Failed to serialize trace event: {e}");
                return;
            }
        };
        line.push('\n');
        if let Err(e) = async {
            let mut file = tokio::fs::OpenOptions::new()
                .append(true)
                .create(true)
                .open(path)
                .await?;
            file.write_all(line.as_bytes()).await?;
            file.flush().await?;
            Ok::<(), std::io::Error>(())
        }
        .await
        {
            warn!("Failed to write trace event: {e}");
        }
    }
}

#[async_trait]
impl EventHandle<EventData<StartPayload>> for TraceLoggingHandler {
    async fn handle(
        &self,
        event: &EventData<StartPayload>,
        conversation: &mut Conversation,
    ) -> anyhow::Result<()> {
        if !self.is_enabled() {
            return Ok(());
        }
        self.write_event(&serde_json::json!({
            "event": "session_start",
            "ts": Utc::now().to_rfc3339(),
            "session_id": conversation.id.into_string(),
            "model": event.model_id.as_str(),
            "agent_id": event.agent.id.as_str(),
        }))
        .await;
        Ok(())
    }
}

#[async_trait]
impl EventHandle<EventData<RequestPayload>> for TraceLoggingHandler {
    async fn handle(
        &self,
        event: &EventData<RequestPayload>,
        conversation: &mut Conversation,
    ) -> anyhow::Result<()> {
        if !self.is_enabled() {
            return Ok(());
        }

        // Record turn and started_at for duration tracking
        {
            let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
            state.turn = event.payload.request_count;
            state.started_at = Some(Instant::now());
        }

        // Only emit context event on turn 1 (request_count == 0)
        if event.payload.request_count == 0 {
            if let Some(context) = &conversation.context {
                let raw_request = crate::dto::openai::Request::from(context.clone())
                    .model(event.model_id.clone());
                // Mirror the same provider transformer pipeline that openai.rs
                // runs at HTTP-send time, so the trace reflects what the LLM
                // actually receives on the wire (e.g. MergeSystemMessages
                // collapsing 2 system messages into 1 for the providers in
                // its predicate). The pipeline's predicates only check
                // `provider.id`; we synthesize a Provider with a placeholder
                // URL solely for trace purposes.
                let provider: Provider<Url> = Provider {
                    id: event.agent.provider.clone(),
                    provider_type: Default::default(),
                    response: None,
                    url: Url::parse("http://placeholder/").unwrap(),
                    models: None,
                    auth_methods: vec![],
                    url_params: vec![],
                    credential: None,
                    custom_headers: None,
                };
                let mut pipeline = ProviderPipeline::new(&provider);
                let request = pipeline.transform(raw_request);
                let request_value = match serde_json::to_value(&request) {
                    Ok(v) => v,
                    Err(e) => {
                        warn!(error = %e, "Failed to serialize for trace");
                        return Ok(());
                    }
                };
                self.write_event(&serde_json::json!({
                    "event": "context",
                    "ts": Utc::now().to_rfc3339(),
                    "messages": request_value.get("messages").cloned().unwrap_or(serde_json::json!([])),
                    "tools": request_value.get("tools").cloned().unwrap_or(serde_json::json!([])),
                }))
                .await;
            }
        }

        Ok(())
    }
}

#[async_trait]
impl EventHandle<EventData<ResponsePayload>> for TraceLoggingHandler {
    async fn handle(
        &self,
        event: &EventData<ResponsePayload>,
        _conversation: &mut Conversation,
    ) -> anyhow::Result<()> {
        if !self.is_enabled() {
            return Ok(());
        }
        let (turn, duration_ms) = {
            let state = self.state.lock().unwrap_or_else(|e| e.into_inner());
            let duration = state.started_at.map(|s| s.elapsed().as_millis() as u64).unwrap_or(0);
            (state.turn, duration)
        };
        let mut val = match serde_json::to_value(&event.payload.message) {
            Ok(v) => v,
            Err(e) => {
                warn!(error = %e, "Failed to serialize for trace");
                return Ok(());
            }
        };
        if let Some(obj) = val.as_object_mut() {
            obj.insert("event".to_string(), serde_json::json!("response"));
            obj.insert("ts".to_string(), serde_json::json!(Utc::now().to_rfc3339()));
            obj.insert("turn".to_string(), serde_json::json!(turn + 1));
            obj.insert("duration_ms".to_string(), serde_json::json!(duration_ms));
        }
        self.write_event(&val).await;
        Ok(())
    }
}

#[async_trait]
impl EventHandle<EventData<ToolcallEndPayload>> for TraceLoggingHandler {
    async fn handle(
        &self,
        event: &EventData<ToolcallEndPayload>,
        _conversation: &mut Conversation,
    ) -> anyhow::Result<()> {
        if !self.is_enabled() {
            return Ok(());
        }
        let turn = {
            let state = self.state.lock().unwrap_or_else(|e| e.into_inner());
            state.turn + 1
        };
        let tool_call = &event.payload.tool_call;
        let result = &event.payload.result;
        self.write_event(&serde_json::json!({
            "event": "tool_result",
            "ts": Utc::now().to_rfc3339(),
            "turn": turn,
            "call_id": tool_call.call_id.as_ref().map(|id| id.as_str()),
            "name": tool_call.name.as_str(),
            "content": result.output.as_str().unwrap_or(""),
            "is_error": result.output.is_error,
        }))
        .await;
        Ok(())
    }
}

#[async_trait]
impl EventHandle<EventData<EndPayload>> for TraceLoggingHandler {
    async fn handle(
        &self,
        _event: &EventData<EndPayload>,
        _conversation: &mut Conversation,
    ) -> anyhow::Result<()> {
        if !self.is_enabled() {
            return Ok(());
        }
        let total_turns = {
            let state = self.state.lock().unwrap_or_else(|e| e.into_inner());
            state.turn + 1
        };
        self.write_event(&serde_json::json!({
            "event": "session_end",
            "ts": Utc::now().to_rfc3339(),
            "total_turns": total_turns,
        }))
        .await;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use forge_domain::{
        Agent, ChatCompletionMessageFull, Context, ContextMessage, FinishReason, ModelId,
        ToolCallFull, ToolCallId, ToolName, ToolResult,
    };
    use pretty_assertions::assert_eq;

    use super::*;

    fn test_agent() -> Agent {
        Agent::new(
            "test-agent",
            "test-provider".to_string().into(),
            ModelId::new("test-model"),
        )
    }

    fn test_model_id() -> ModelId {
        ModelId::new("test-model")
    }

    #[tokio::test]
    async fn test_disabled_handler_is_noop() {
        let handler = TraceLoggingHandler::with_path(None);
        assert_eq!(handler.is_enabled(), false);

        let mut conversation = Conversation::generate();
        let event = EventData::new(test_agent(), test_model_id(), StartPayload);
        let result = handler.handle(&event, &mut conversation).await;
        assert!(result.is_ok());
    }

    fn build_context() -> Context {
        Context::default()
            .add_message(ContextMessage::system("You are a helpful assistant"))
            .add_message(ContextMessage::user("Hello", None))
    }

    #[tokio::test]
    async fn test_session_start_event() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("trace.jsonl");
        let handler = TraceLoggingHandler::with_path(Some(path.clone()));

        let mut conversation = Conversation::generate();
        let event = EventData::new(test_agent(), test_model_id(), StartPayload);
        handler.handle(&event, &mut conversation).await.unwrap();

        let content = std::fs::read_to_string(&path).unwrap();
        let val: serde_json::Value = serde_json::from_str(content.trim()).unwrap();
        assert_eq!(val["event"], "session_start");
        assert_eq!(val["session_id"], conversation.id.into_string());
        assert_eq!(val["model"], "test-model");
        assert_eq!(val["agent_id"], "test-agent");
        assert!(val["ts"].is_string());
    }

    #[tokio::test]
    async fn test_context_event_emitted_on_turn_1() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("trace.jsonl");
        let handler = TraceLoggingHandler::with_path(Some(path.clone()));

        let context = build_context();
        let mut conversation = Conversation::generate().context(Some(context));
        let event = EventData::new(test_agent(), test_model_id(), RequestPayload::new(0));
        handler.handle(&event, &mut conversation).await.unwrap();

        let content = std::fs::read_to_string(&path).unwrap();
        let val: serde_json::Value = serde_json::from_str(content.trim()).unwrap();
        assert_eq!(val["event"], "context");
        assert!(val["messages"].is_array());
        assert!(val["messages"].as_array().unwrap().len() >= 2);
        assert!(val["ts"].is_string());
    }

    #[tokio::test]
    async fn test_context_event_not_emitted_on_turn_2() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("trace.jsonl");
        let handler = TraceLoggingHandler::with_path(Some(path.clone()));

        let context = build_context();
        let mut conversation = Conversation::generate().context(Some(context));
        let event = EventData::new(test_agent(), test_model_id(), RequestPayload::new(1));
        handler.handle(&event, &mut conversation).await.unwrap();

        assert!(!path.exists());
    }

    #[tokio::test]
    async fn test_response_event() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("trace.jsonl");
        let handler = TraceLoggingHandler::with_path(Some(path.clone()));

        // Simulate that we're on turn 0 (first request)
        {
            let mut state = handler.state.lock().unwrap();
            state.turn = 0;
            state.started_at = Some(Instant::now());
        }

        let message = ChatCompletionMessageFull {
            content: "Hello world".to_string(),
            thought_signature: None,
            reasoning: None,
            reasoning_details: None,
            tool_calls: vec![],
            usage: Default::default(),
            finish_reason: Some(FinishReason::Stop),
            phase: None,
        };
        let mut conversation = Conversation::generate();
        let event = EventData::new(test_agent(), test_model_id(), ResponsePayload::new(message));
        handler.handle(&event, &mut conversation).await.unwrap();

        let content = std::fs::read_to_string(&path).unwrap();
        let val: serde_json::Value = serde_json::from_str(content.trim()).unwrap();
        assert_eq!(val["event"], "response");
        assert_eq!(val["turn"], 1);
        assert_eq!(val["content"], "Hello world");
        assert_eq!(val["finish_reason"], "Stop");
        assert!(val["duration_ms"].is_number());
        assert!(val["ts"].is_string());
    }

    #[tokio::test]
    async fn test_tool_result_event() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("trace.jsonl");
        let handler = TraceLoggingHandler::with_path(Some(path.clone()));

        // Set turn state
        {
            let mut state = handler.state.lock().unwrap();
            state.turn = 0;
        }

        let tool_call = ToolCallFull {
            name: ToolName::from("bash"),
            call_id: Some(ToolCallId::new("c1")),
            arguments: serde_json::json!({}).into(),
            thought_signature: None,
        };
        let result = ToolResult::new(ToolName::from("bash"))
            .call_id(ToolCallId::new("c1"))
            .success("output text");

        let mut conversation = Conversation::generate();
        let event = EventData::new(
            test_agent(),
            test_model_id(),
            ToolcallEndPayload::new(tool_call, result),
        );
        handler.handle(&event, &mut conversation).await.unwrap();

        let content = std::fs::read_to_string(&path).unwrap();
        let val: serde_json::Value = serde_json::from_str(content.trim()).unwrap();
        assert_eq!(val["event"], "tool_result");
        assert_eq!(val["turn"], 1);
        assert_eq!(val["call_id"], "c1");
        assert_eq!(val["name"], "bash");
        assert_eq!(val["content"], "output text");
        assert_eq!(val["is_error"], false);
        assert!(val["ts"].is_string());
    }

    #[tokio::test]
    async fn test_session_end_event() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("trace.jsonl");
        let handler = TraceLoggingHandler::with_path(Some(path.clone()));

        // Set turn=2 so total_turns should be 3
        {
            let mut state = handler.state.lock().unwrap();
            state.turn = 2;
        }

        let mut conversation = Conversation::generate();
        let event = EventData::new(test_agent(), test_model_id(), EndPayload);
        handler.handle(&event, &mut conversation).await.unwrap();

        let content = std::fs::read_to_string(&path).unwrap();
        let val: serde_json::Value = serde_json::from_str(content.trim()).unwrap();
        assert_eq!(val["event"], "session_end");
        assert_eq!(val["total_turns"], 3);
        assert!(val["ts"].is_string());
    }

    #[tokio::test]
    async fn test_full_session_event_ordering() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("trace.jsonl");
        let handler = TraceLoggingHandler::with_path(Some(path.clone()));

        let context = build_context();
        let mut conversation = Conversation::generate().context(Some(context));

        // 1. session_start
        let start_event = EventData::new(test_agent(), test_model_id(), StartPayload);
        handler.handle(&start_event, &mut conversation).await.unwrap();

        // 2. context (RequestPayload with request_count=0)
        let req_event = EventData::new(test_agent(), test_model_id(), RequestPayload::new(0));
        handler.handle(&req_event, &mut conversation).await.unwrap();

        // 3. response with tool_calls (turn 1)
        let tool_call = ToolCallFull {
            name: ToolName::from("bash"),
            call_id: Some(ToolCallId::new("c1")),
            arguments: serde_json::json!({}).into(),
            thought_signature: None,
        };
        let message_with_tool = ChatCompletionMessageFull {
            content: "Let me run that".to_string(),
            thought_signature: None,
            reasoning: None,
            reasoning_details: None,
            tool_calls: vec![tool_call.clone()],
            usage: Default::default(),
            finish_reason: Some(FinishReason::ToolCalls),
            phase: None,
        };
        let resp_event =
            EventData::new(test_agent(), test_model_id(), ResponsePayload::new(message_with_tool));
        handler.handle(&resp_event, &mut conversation).await.unwrap();

        // 4. tool_result
        let tool_result = ToolResult::new(ToolName::from("bash"))
            .call_id(ToolCallId::new("c1"))
            .success("command output");
        let toolcall_end_event = EventData::new(
            test_agent(),
            test_model_id(),
            ToolcallEndPayload::new(tool_call, tool_result),
        );
        handler
            .handle(&toolcall_end_event, &mut conversation)
            .await
            .unwrap();

        // 5. RequestPayload with request_count=1 (no context emitted)
        let req2_event = EventData::new(test_agent(), test_model_id(), RequestPayload::new(1));
        handler.handle(&req2_event, &mut conversation).await.unwrap();

        // 6. response without tool_calls (turn 2, finish_reason=Stop)
        let final_message = ChatCompletionMessageFull {
            content: "Done!".to_string(),
            thought_signature: None,
            reasoning: None,
            reasoning_details: None,
            tool_calls: vec![],
            usage: Default::default(),
            finish_reason: Some(FinishReason::Stop),
            phase: None,
        };
        let resp2_event =
            EventData::new(test_agent(), test_model_id(), ResponsePayload::new(final_message));
        handler
            .handle(&resp2_event, &mut conversation)
            .await
            .unwrap();

        // 7. session_end
        let end_event = EventData::new(test_agent(), test_model_id(), EndPayload);
        handler.handle(&end_event, &mut conversation).await.unwrap();

        // Verify: 6 lines in file, correct event types in order, correct turn numbers
        let content = std::fs::read_to_string(&path).unwrap();
        let lines: Vec<&str> = content.trim().lines().collect();
        assert_eq!(lines.len(), 6);

        let events: Vec<serde_json::Value> = lines
            .iter()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();

        assert_eq!(events[0]["event"], "session_start");
        assert_eq!(events[1]["event"], "context");
        assert_eq!(events[2]["event"], "response");
        assert_eq!(events[2]["turn"], 1);
        assert_eq!(events[3]["event"], "tool_result");
        assert_eq!(events[3]["turn"], 1);
        assert_eq!(events[4]["event"], "response");
        assert_eq!(events[4]["turn"], 2);
        assert_eq!(events[5]["event"], "session_end");
        assert_eq!(events[5]["total_turns"], 2);
    }
}
