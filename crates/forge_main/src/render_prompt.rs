use std::path::PathBuf;

use anyhow::Result;
use forge_api::{API, ForgeAPI};
use forge_config::ForgeConfig;
use forge_domain::AgentId;

/// Implements `forge prompt render`. Builds a fresh [`ForgeAPI`] rooted at
/// the resolved cwd, asks it to render the agent's system prompt, and
/// prints `block1 + "\n\n" + block2` to stdout — the same merged shape
/// `MergeSystemMessages` produces for `OPENAI_COMPATIBLE` providers.
///
/// The cwd is resolved as: subcommand `--cwd` if provided, otherwise the
/// `default_cwd` computed by `main.rs` (the top-level `-C/--directory`
/// flag, or the process cwd).
pub async fn run(
    directory: Option<PathBuf>,
    agent: Option<AgentId>,
    config: ForgeConfig,
    default_cwd: PathBuf,
) -> Result<()> {
    let cwd = match directory {
        Some(path) => path
            .canonicalize()
            .map_err(|err| anyhow::anyhow!("invalid --cwd {}: {err}", path.display()))?,
        None => default_cwd,
    };

    let api = ForgeAPI::init(cwd, config);

    let agent_id = match agent {
        Some(id) => id,
        None => api
            .get_active_agent()
            .await
            .unwrap_or_else(|| AgentId::new("forge")),
    };

    match api.render_system_prompt(agent_id.clone()).await? {
        Some((block_1, block_2)) => {
            print!("{block_1}\n\n{block_2}");
            Ok(())
        }
        None => Err(anyhow::anyhow!(
            "agent '{}' has no system_prompt template",
            agent_id.as_str()
        )),
    }
}
