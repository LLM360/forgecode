use std::io::{self, Read};

use anyhow::{Context, Result};
use forge_domain::{ToolCallFull, ToolCatalog};

fn normalize_tool_call(tool_call: ToolCallFull) -> ToolCallFull {
    if !ToolCatalog::contains(&tool_call.name) {
        return tool_call;
    }

    let call_id = tool_call.call_id.clone();
    let thought_signature = tool_call.thought_signature.clone();

    match ToolCatalog::try_from(tool_call.clone()) {
        Ok(tool_input) => {
            let mut normalized = ToolCallFull::from(tool_input);
            normalized.call_id = call_id;
            normalized.thought_signature = thought_signature;
            normalized
        }
        Err(_) => tool_call,
    }
}

fn main() -> Result<()> {
    let mut input = String::new();
    io::stdin()
        .read_to_string(&mut input)
        .context("failed to read tool call JSON from stdin")?;

    let tool_call: ToolCallFull =
        serde_json::from_str(&input).context("stdin must be one Forge tool-call JSON object")?;
    let normalized = normalize_tool_call(tool_call);

    serde_json::to_writer_pretty(io::stdout(), &normalized)
        .context("failed to write normalized tool call JSON")?;
    println!();

    Ok(())
}

#[cfg(test)]
mod tests {
    use forge_domain::ToolCallFull;
    use serde_json::json;

    use super::normalize_tool_call;

    #[test]
    fn normalizes_stringified_valid_arguments() {
        let input: ToolCallFull = serde_json::from_value(json!({
            "name": "write",
            "call_id": "functions.write:1",
            "arguments": "{\"file_path\":\"/tmp/out.txt\",\"content\":\"hello\"}"
        }))
        .unwrap();

        let actual = serde_json::to_value(normalize_tool_call(input)).unwrap();

        assert_eq!(actual["call_id"], "functions.write:1");
        assert_eq!(actual["arguments"]["file_path"], "/tmp/out.txt");
        assert_eq!(actual["arguments"]["content"], "hello");
    }

    #[test]
    fn preserves_unrecoverable_malformed_arguments() {
        let input: ToolCallFull = serde_json::from_value(json!({
            "name": "shell",
            "call_id": "functions.shell:1",
            "arguments": "{\"command\":\"touch \"/tmp/.done\"\"}"
        }))
        .unwrap();

        let actual = serde_json::to_value(normalize_tool_call(input)).unwrap();

        assert_eq!(actual["call_id"], "functions.shell:1");
        assert!(actual["arguments"].is_string());
    }
}
