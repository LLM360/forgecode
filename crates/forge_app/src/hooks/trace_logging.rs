use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use async_trait::async_trait;
use chrono::Utc;
use forge_domain::{
    Conversation, EndPayload, EventData, EventHandle, Provider, ResponsePayload, StartPayload,
    ToolcallEndPayload, Transformer,
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

    /// Writes a per-turn `request` event capturing the full post-pipeline
    /// request body sent to the LLM, after the orch transformer chain has run
    /// and after `ProviderPipeline` has been applied. Called from
    /// `Orchestrator::execute_chat_turn` between the transformer chain and the
    /// `chat_agent` HTTP call. De-duplicates so retries via `retry_with_config`
    /// emit exactly one event per logical turn.
    pub async fn write_request_event(
        &self,
        transformed_context: &forge_domain::Context,
        provider_id: &forge_domain::ProviderId,
        model_id: &forge_domain::ModelId,
        turn: usize,
    ) {
        if !self.is_enabled() {
            return;
        }

        {
            let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
            let new_turn = turn - 1;
            if state.turn == new_turn && state.started_at.is_some() {
                return;
            }
            state.turn = new_turn;
            state.started_at = Some(Instant::now());
        }

        let raw_request = crate::dto::openai::Request::from(transformed_context.clone())
            .model(model_id.clone());

        let provider: Provider<Url> = Provider {
            id: provider_id.clone(),
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

        let body = match serde_json::to_value(&request) {
            Ok(v) => v,
            Err(e) => {
                warn!(error = %e, "Failed to serialize for trace");
                return;
            }
        };

        self.write_event(&serde_json::json!({
            "event": "request",
            "ts": Utc::now().to_rfc3339(),
            "turn": turn,
            "body": body,
        }))
        .await;
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
        Agent, ChatCompletionMessageFull, FinishReason, ModelId, ToolCallFull, ToolCallId,
        ToolName, ToolResult,
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
    async fn test_response_event() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("trace.jsonl");
        let handler = TraceLoggingHandler::with_path(Some(path.clone()));

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
}
