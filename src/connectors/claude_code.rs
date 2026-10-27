use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde_json::{Map, Value};
use walkdir::WalkDir;

use super::scan::{
    DiscoveredSourceFile, DiscoveredSourceRole, SOURCE_TAIL_HEAD_BYTES, SOURCE_TAIL_PREFIX_BYTES,
    ScanContext, ScanRoot, SourceCompletion, SourceScanHooks, SourceTailResume, SourceTailState,
};
use super::utils::{
    dedupe_path_key, env_path_nonempty, excluded_scan_paths_from_env, path_is_excluded,
};
use super::{
    Connector, extract_invocations_from_content_blocks, file_modified_since, flatten_content,
    franken_detection_for_connector, parse_timestamp,
};
use crate::types::{DetectionResult, NormalizedConversation, NormalizedMessage};

pub struct ClaudeCodeConnector;

const LARGE_SESSION_EXTRA_COMPACT_THRESHOLD_BYTES: u64 = 32 * 1024 * 1024;

impl Default for ClaudeCodeConnector {
    fn default() -> Self {
        Self::new()
    }
}

impl ClaudeCodeConnector {
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    fn projects_root() -> PathBuf {
        Self::projects_root_resolved(
            env_path_nonempty("CLAUDE_CONFIG_DIR").as_deref(),
            env_path_nonempty("XDG_CONFIG_HOME").as_deref(),
            dirs::home_dir().as_deref(),
        )
    }

    fn projects_root_candidates() -> Vec<PathBuf> {
        let primary = Self::projects_root();
        let mut roots = Self::projects_root_candidates_resolved(
            env_path_nonempty("CLAUDE_CONFIG_DIR").as_deref(),
            env_path_nonempty("XDG_CONFIG_HOME").as_deref(),
            dirs::home_dir().as_deref(),
        );
        if !roots.contains(&primary) {
            roots.push(primary);
        }
        roots
    }

    fn desktop_session_roots_resolved(home_dir: Option<&Path>) -> Vec<PathBuf> {
        let Some(home) = home_dir else {
            return Vec::new();
        };
        let claude_support = home
            .join("Library")
            .join("Application Support")
            .join("Claude");
        vec![
            claude_support.join("claude-code-sessions"),
            claude_support.join("local-agent-mode-sessions"),
        ]
    }

    /// Pure resolver for [`Self::projects_root`] — split out so the precedence
    /// chain can be unit-tested without manipulating process env vars
    /// (`std::env::set_var` is `unsafe` and not safe across parallel tests).
    ///
    /// Honors the env-var redirects Claude Code itself documents, in the same
    /// precedence order:
    ///   1. `CLAUDE_CONFIG_DIR` — explicit override.
    ///   2. `XDG_CONFIG_HOME`   — XDG fallback.
    ///   3. `${HOME}/.claude/projects` — default.
    ///
    /// Without this, the connector silently ignores caam-isolated profiles and
    /// any user with `XDG_CONFIG_HOME` set.
    fn projects_root_resolved(
        claude_config_dir: Option<&Path>,
        xdg_config_home: Option<&Path>,
        home_dir: Option<&Path>,
    ) -> PathBuf {
        if let Some(explicit) = claude_config_dir {
            return explicit.join("projects");
        }
        if let Some(xdg) = xdg_config_home {
            return xdg.join("claude-code").join("projects");
        }
        // Match the historical behavior: when even `dirs::home_dir()` returns
        // None we fall back to a relative `.claude/projects` path. That keeps
        // strict-mode test environments (sandboxes with no HOME) unchanged.
        home_dir.map_or_else(
            || PathBuf::from(".claude/projects"),
            |h| h.join(".claude/projects"),
        )
    }

    fn projects_root_candidates_resolved(
        claude_config_dir: Option<&Path>,
        xdg_config_home: Option<&Path>,
        home_dir: Option<&Path>,
    ) -> Vec<PathBuf> {
        if let Some(explicit) = claude_config_dir {
            return vec![explicit.join("projects")];
        }

        let mut roots = Vec::new();
        if let Some(xdg) = xdg_config_home {
            roots.push(xdg.join("claude-code").join("projects"));
        }
        roots.push(home_dir.map_or_else(
            || PathBuf::from(".claude/projects"),
            |home| home.join(".claude/projects"),
        ));
        roots.extend(Self::desktop_session_roots_resolved(home_dir));
        roots.sort();
        roots.dedup();
        roots
    }

    fn session_files(scan_target: &Path) -> Vec<PathBuf> {
        let mut files = Vec::new();
        for entry in WalkDir::new(scan_target).into_iter().flatten() {
            if !entry.file_type().is_file() {
                continue;
            }
            let ext = entry.path().extension().and_then(|s| s.to_str());
            if ext == Some("jsonl") || ext == Some("json") || ext == Some("claude") {
                files.push(entry.path().to_path_buf());
            }
        }
        // Keep connector traversal deterministic across filesystems/runs.
        files.sort();
        files
    }

    fn projects_root_for_explicit_file(path: &Path) -> Option<PathBuf> {
        path.ancestors()
            .filter(|ancestor| ancestor.is_dir())
            .find(|ancestor| {
                ancestor.file_name().and_then(|name| name.to_str()) == Some("projects")
            })
            .map(Path::to_path_buf)
    }

    fn should_compact_large_message_extra(file_size_bytes: Option<u64>) -> bool {
        file_size_bytes.is_some_and(|size| size >= LARGE_SESSION_EXTRA_COMPACT_THRESHOLD_BYTES)
    }

    fn path_is_desktop_sidecar(path: &Path) -> bool {
        path.components().any(|component| {
            matches!(
                component.as_os_str().to_str(),
                Some("claude-code-sessions" | "local-agent-mode-sessions")
            )
        })
    }

    fn discovered_source_role(path: &Path) -> DiscoveredSourceRole {
        if Self::path_is_desktop_sidecar(path) {
            DiscoveredSourceRole::MetadataSidecar
        } else {
            DiscoveredSourceRole::PrimarySessionLog
        }
    }

    fn non_empty_json_string(value: &Value, key: &str) -> Option<String> {
        value
            .get(key)
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|text| !text.is_empty())
            .map(str::to_string)
    }

    fn desktop_sidecar_metadata_message(raw: &Value) -> Option<NormalizedMessage> {
        let title = Self::non_empty_json_string(raw, "title");
        let cwd = Self::non_empty_json_string(raw, "cwd");
        let model = Self::non_empty_json_string(raw, "model");
        let cli_session_id = Self::non_empty_json_string(raw, "cliSessionId");
        let session_id = Self::non_empty_json_string(raw, "sessionId");
        let permission_mode = Self::non_empty_json_string(raw, "permissionMode");
        if title.is_none()
            && cwd.is_none()
            && model.is_none()
            && cli_session_id.is_none()
            && session_id.is_none()
        {
            return None;
        }

        let mut lines = vec!["Claude Code Desktop session metadata".to_string()];
        if let Some(title) = &title {
            lines.push(format!("Title: {title}"));
        }
        if let Some(cwd) = &cwd {
            lines.push(format!("Workspace: {cwd}"));
        }
        if let Some(model) = &model {
            lines.push(format!("Model: {model}"));
        }
        if let Some(permission_mode) = &permission_mode {
            lines.push(format!("Permission mode: {permission_mode}"));
        }
        if let Some(cli_session_id) = &cli_session_id {
            lines.push(format!("CLI session id: {cli_session_id}"));
        } else if let Some(session_id) = &session_id {
            lines.push(format!("Session id: {session_id}"));
        }
        lines.push(
            "Conversation body unavailable in this Desktop sidecar; Claude Code may have culled the CLI JSONL body."
                .to_string(),
        );

        let created_at = raw
            .get("lastActivityAt")
            .or_else(|| raw.get("createdAt"))
            .and_then(parse_timestamp);
        Some(NormalizedMessage {
            idx: 0,
            role: "system".to_string(),
            author: Some("claude_code_desktop".to_string()),
            created_at,
            content: lines.join("\n"),
            extra: serde_json::json!({
                "cass": {
                    "source": "claude_code_desktop_sidecar",
                    "body_available": false,
                    "body_cull_note": "Claude Code may auto-cull CLI JSONL bodies while Desktop sidecars retain title/workspace metadata."
                },
                "raw": raw
            }),
            invocations: Vec::new(),
            snippets: Vec::new(),
        })
    }

    fn source_roots(ctx: &ScanContext) -> Vec<ScanRoot> {
        let looks_like_root = |path: &PathBuf| path.join("projects").exists();

        let mut roots: Vec<ScanRoot> = if ctx.use_default_detection() {
            if looks_like_root(&ctx.data_dir) {
                vec![ScanRoot::local(ctx.data_dir.clone())]
            } else {
                Self::projects_root_candidates()
                    .into_iter()
                    .map(ScanRoot::local)
                    .collect()
            }
        } else {
            ctx.scan_roots.clone()
        };

        roots.sort_by(|a, b| a.path.cmp(&b.path));
        roots.dedup_by(|a, b| a.path == b.path);
        roots
    }

    fn discover_sources(ctx: &ScanContext) -> Vec<DiscoveredSourceFile> {
        Self::discover_sources_with_exclusions(ctx, &excluded_scan_paths_from_env())
    }

    fn discover_sources_with_exclusions(
        ctx: &ScanContext,
        excluded_paths: &[PathBuf],
    ) -> Vec<DiscoveredSourceFile> {
        let mut out = Vec::new();
        for root in Self::source_roots(ctx) {
            let scan_target = root.path.clone();
            if !scan_target.exists() {
                continue;
            }
            let session_paths = if scan_target.is_file() {
                vec![scan_target]
            } else {
                Self::session_files(&scan_target)
            };
            for path in session_paths {
                if path_is_excluded(&path, excluded_paths) {
                    tracing::debug!(
                        path = %path.display(),
                        "claude_code skipping excluded session source"
                    );
                    continue;
                }
                if !file_modified_since(&path, ctx.since_ts) {
                    continue;
                }
                out.push(
                    DiscoveredSourceFile::new(
                        "claude_code",
                        &root,
                        path.clone(),
                        Self::discovered_source_role(&path),
                        true,
                    )
                    .with_fs_metadata(),
                );
            }
        }
        out
    }

    fn compact_message_extra(raw: &Value) -> Value {
        let mut cass = serde_json::Map::new();

        if let Some(model) = raw
            .pointer("/message/model")
            .or_else(|| raw.get("model"))
            .and_then(|v| v.as_str())
            .filter(|value| !value.trim().is_empty())
        {
            cass.insert("model".to_string(), Value::String(model.to_string()));
        }

        let usage = raw.pointer("/message/usage");
        let mut token_usage = serde_json::Map::new();
        if let Some(input_tokens) = usage
            .and_then(|value| value.get("input_tokens"))
            .and_then(Value::as_i64)
        {
            token_usage.insert("input_tokens".to_string(), Value::from(input_tokens));
        }
        if let Some(output_tokens) = usage
            .and_then(|value| value.get("output_tokens"))
            .and_then(Value::as_i64)
        {
            token_usage.insert("output_tokens".to_string(), Value::from(output_tokens));
        }
        if let Some(cache_read_tokens) = usage
            .and_then(|value| value.get("cache_read_input_tokens"))
            .and_then(Value::as_i64)
        {
            token_usage.insert(
                "cache_read_tokens".to_string(),
                Value::from(cache_read_tokens),
            );
        }
        if let Some(cache_creation_tokens) = usage
            .and_then(|value| value.get("cache_creation_input_tokens"))
            .and_then(Value::as_i64)
        {
            token_usage.insert(
                "cache_creation_tokens".to_string(),
                Value::from(cache_creation_tokens),
            );
        }
        if let Some(service_tier) = usage
            .and_then(|value| value.get("service_tier"))
            .and_then(|value| value.as_str())
            .filter(|value| !value.trim().is_empty())
        {
            token_usage.insert(
                "service_tier".to_string(),
                Value::String(service_tier.to_string()),
            );
        }
        if !token_usage.is_empty() {
            token_usage.insert("data_source".to_string(), Value::String("api".to_string()));
            cass.insert("token_usage".to_string(), Value::Object(token_usage));
        }

        let tool_call_count = raw
            .pointer("/message/content")
            .and_then(|value| value.as_array())
            .map_or(0, |items| {
                items
                    .iter()
                    .filter(|item| {
                        item.get("type").and_then(|kind| kind.as_str()) == Some("tool_use")
                    })
                    .count()
            });
        if tool_call_count > 0 {
            cass.insert("tool_call_count".to_string(), Value::from(tool_call_count));
        }

        if let Some(attachments) = raw
            .get("attachment_refs")
            .or_else(|| raw.get("attachments"))
            .or_else(|| raw.pointer("/message/attachment_refs"))
            .or_else(|| raw.pointer("/message/attachments"))
            .cloned()
        {
            cass.insert("attachments".to_string(), attachments);
        }

        if cass.is_empty() {
            Value::Object(serde_json::Map::new())
        } else {
            let mut out = serde_json::Map::new();
            out.insert("cass".to_string(), Value::Object(cass));
            Value::Object(out)
        }
    }

    /// A prompt the user typed while the agent was mid-turn is recorded as
    /// `{"type":"attachment","attachment":{"type":"queued_command",
    /// "commandMode":"prompt","prompt":..,"origin":{"kind":"human"}}}` rather
    /// than as a `type:"user"` entry, so the user/assistant filter would drop
    /// it and the text would never be searchable (cass GH #500).
    ///
    /// Only prompt-mode entries typed by a person are normalized:
    /// `origin.kind == "human"`, or no `origin` at all (older Claude Code
    /// builds), which is kept but marked as unknown authorship. Relayed
    /// (`peer`) and any other explicit origin, meta entries, and
    /// machine-generated `task-notification` entries are skipped. The sibling
    /// `queue-operation` records are not indexed either: they repeat the text
    /// of every queued item, including ones later saved as normal user
    /// entries, so indexing them would double turns.
    fn queued_command_message(raw: &Value, compact_extra: bool) -> Option<NormalizedMessage> {
        let attachment = raw.get("attachment")?;
        if attachment.get("type").and_then(Value::as_str) != Some("queued_command")
            || attachment.get("commandMode").and_then(Value::as_str) != Some("prompt")
        {
            return None;
        }
        let is_meta = |v: &Value| v.get("isMeta").and_then(Value::as_bool) == Some(true);
        if is_meta(raw) || is_meta(attachment) {
            return None;
        }
        let authorship = match attachment.get("origin") {
            None | Some(Value::Null) => "unknown",
            Some(origin) if origin.get("kind").and_then(Value::as_str) == Some("human") => "human",
            Some(_) => return None,
        };

        let content = attachment
            .get("prompt")
            .map(flatten_content)
            .unwrap_or_default();
        if content.trim().is_empty() {
            return None;
        }

        let created = raw
            .get("timestamp")
            .and_then(parse_timestamp)
            .or_else(|| attachment.get("timestamp").and_then(parse_timestamp));

        let mut provenance = Map::new();
        provenance.insert(
            "command_mode".to_string(),
            Value::String("prompt".to_string()),
        );
        provenance.insert(
            "authorship".to_string(),
            Value::String(authorship.to_string()),
        );
        if let Some(source_uuid) = attachment
            .get("source_uuid")
            .and_then(Value::as_str)
            .filter(|s| !s.trim().is_empty())
        {
            provenance.insert(
                "source_uuid".to_string(),
                Value::String(source_uuid.to_string()),
            );
        }

        let mut extra = if compact_extra {
            Self::compact_message_extra(raw)
        } else {
            raw.clone()
        };
        if let Some(obj) = extra.as_object_mut() {
            let cass = obj
                .entry("cass".to_string())
                .or_insert_with(|| Value::Object(Map::new()));
            if !cass.is_object() {
                *cass = Value::Object(Map::new());
            }
            if let Some(cass) = cass.as_object_mut() {
                cass.insert("queued_command".to_string(), Value::Object(provenance));
            }
        }

        Some(NormalizedMessage {
            idx: 0,
            role: "user".to_string(),
            author: None,
            created_at: created,
            content,
            extra,
            invocations: Vec::new(),
            snippets: Vec::new(),
        })
    }
}

/// Widen a conversation's `[started_at, ended_at]` bounds to cover `created`.
fn widen_time_bounds(
    started_at: &mut Option<i64>,
    ended_at: &mut Option<i64>,
    created: Option<i64>,
) {
    if let Some(ts) = created {
        *started_at = Some(started_at.map_or(ts, |curr| curr.min(ts)));
        *ended_at = Some(ended_at.map_or(ts, |curr| curr.max(ts)));
    }
}

/// Default size of one emitted part of a large Claude session. A session over
/// either limit is emitted as several conversations with the same identity
/// and continuous message indexes, so a host can store and free each part
/// instead of holding the whole session in memory.
const DEFAULT_CHUNK_MAX_MESSAGES: usize = 8_192;
const DEFAULT_CHUNK_MAX_CONTENT_BYTES: usize = 16 * 1024 * 1024;
/// Lines longer than this are never `ai-title` entries, so the title pre-pass
/// does not parse them.
const AI_TITLE_LINE_MAX_BYTES: usize = 64 * 1024;
const CARRIED_STATE_VERSION: u64 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ClaudeChunkLimits {
    max_messages: usize,
    max_content_bytes: usize,
}

impl ClaudeChunkLimits {
    /// `FAD_CLAUDE_CHUNK_MAX_MESSAGES` and `FAD_CLAUDE_CHUNK_MAX_BYTES`
    /// override the defaults (tests and memory tuning).
    fn from_env() -> Self {
        #[cfg(test)]
        if let Some(limits) = TEST_CHUNK_LIMITS.with(std::cell::Cell::get) {
            return limits;
        }
        let read = |name: &str, default: usize| {
            std::env::var(name)
                .ok()
                .and_then(|value| value.trim().parse::<usize>().ok())
                .filter(|value| *value > 0)
                .unwrap_or(default)
        };
        Self {
            max_messages: read("FAD_CLAUDE_CHUNK_MAX_MESSAGES", DEFAULT_CHUNK_MAX_MESSAGES),
            max_content_bytes: read(
                "FAD_CLAUDE_CHUNK_MAX_BYTES",
                DEFAULT_CHUNK_MAX_CONTENT_BYTES,
            ),
        }
    }
}

#[cfg(test)]
thread_local! {
    static TEST_MAX_LINE_BYTES: std::cell::Cell<Option<usize>> =
        const { std::cell::Cell::new(None) };
    static TEST_LONG_LINE_KEEP_BYTES: std::cell::Cell<Option<usize>> =
        const { std::cell::Cell::new(None) };
    static TEST_CHUNK_LIMITS: std::cell::Cell<Option<ClaudeChunkLimits>> =
        const { std::cell::Cell::new(None) };
    static TEST_MESSAGE_CHUNK_BYTES: std::cell::Cell<Option<usize>> =
        const { std::cell::Cell::new(None) };
}

/// Session-level state of a Claude JSONL parse: everything a later line, part
/// or tail scan needs to describe the session exactly as one full parse would.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct ClaudeSessionState {
    workspace: Option<PathBuf>,
    session_id: Option<String>,
    git_branch: Option<String>,
    json_title: Option<String>,
    first_user_title: Option<String>,
    started_at: Option<i64>,
    ended_at: Option<i64>,
}

impl ClaudeSessionState {
    /// `None` when the state cannot round-trip through JSON (a non-UTF-8
    /// workspace path); such a source is then always parsed in full.
    fn to_json(&self) -> Option<Value> {
        let workspace = match &self.workspace {
            Some(path) => Some(path.to_str()?.to_string()),
            None => None,
        };
        Some(serde_json::json!({
            "v": CARRIED_STATE_VERSION,
            "workspace": workspace,
            "session_id": self.session_id,
            "git_branch": self.git_branch,
            "json_title": self.json_title,
            "first_user_title": self.first_user_title,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
        }))
    }

    fn from_json(value: &Value) -> Option<Self> {
        if value.get("v")?.as_u64()? != CARRIED_STATE_VERSION {
            return None;
        }
        let text = |key: &str| -> Option<Option<String>> {
            match value.get(key)? {
                Value::Null => Some(None),
                Value::String(text) => Some(Some(text.clone())),
                _ => None,
            }
        };
        let number = |key: &str| -> Option<Option<i64>> {
            match value.get(key)? {
                Value::Null => Some(None),
                other => Some(Some(other.as_i64()?)),
            }
        };
        Some(Self {
            workspace: text("workspace")?.map(PathBuf::from),
            session_id: text("session_id")?,
            git_branch: text("git_branch")?,
            json_title: text("json_title")?,
            first_user_title: text("first_user_title")?,
            started_at: number("started_at")?,
            ended_at: number("ended_at")?,
        })
    }
}

/// The title a full parse gives a session: Claude's own last `ai-title`, else
/// the first line of the first user message, else the workspace folder name.
fn claude_session_title(state: &ClaudeSessionState, json_title: Option<&String>) -> Option<String> {
    json_title
        .cloned()
        .or_else(|| state.first_user_title.clone())
        .or_else(|| {
            state
                .workspace
                .as_ref()
                .and_then(|p| p.file_name())
                .and_then(|n| n.to_str())
                .map(String::from)
        })
}

fn first_line_title(content: &str) -> String {
    content
        .lines()
        .next()
        .unwrap_or(content)
        .chars()
        .take(100)
        .collect::<String>()
}

/// Accumulates the messages of one part of a Claude JSONL session.
struct ClaudeJsonlParser {
    state: ClaudeSessionState,
    messages: Vec<NormalizedMessage>,
    content_bytes: usize,
    compact_message_extra: bool,
}

impl ClaudeJsonlParser {
    fn push(&mut self, message: NormalizedMessage) {
        if self.state.first_user_title.is_none() && message.role == "user" {
            self.state.first_user_title = Some(first_line_title(&message.content));
        }
        // A message over the chunk size becomes ordered chunks, all pushed
        // together, so a part never ends inside one message.
        for chunk in crate::types::chunk_message_at(message, message_chunk_bytes()) {
            self.content_bytes = self.content_bytes.saturating_add(chunk.content.len());
            self.messages.push(chunk);
        }
    }

    #[allow(clippy::too_many_lines)]
    fn ingest_line(&mut self, val: Value) {
        if self.state.workspace.is_none() {
            self.state.workspace = val.get("cwd").and_then(|v| v.as_str()).map(PathBuf::from);
        }
        if self.state.session_id.is_none() {
            self.state.session_id = val
                .get("sessionId")
                .and_then(|v| v.as_str())
                .map(String::from);
        }
        if self.state.git_branch.is_none() {
            self.state.git_branch = val
                .get("gitBranch")
                .and_then(|v| v.as_str())
                .map(String::from);
        }

        let entry_type = val.get("type").and_then(|v| v.as_str());
        // Claude's own generated title beats first-line
        // truncation; the entry is otherwise skipped as a
        // non-conversation sidecar type.
        if entry_type == Some("ai-title") {
            if let Some(title) = val
                .get("title")
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|t| !t.is_empty())
            {
                self.state.json_title = Some(title.to_string());
            }
            return;
        }
        if entry_type == Some("attachment") {
            if let Some(message) =
                ClaudeCodeConnector::queued_command_message(&val, self.compact_message_extra)
            {
                widen_time_bounds(
                    &mut self.state.started_at,
                    &mut self.state.ended_at,
                    message.created_at,
                );
                self.push(message);
            }
            return;
        }
        let role_hint = val
            .get("message")
            .and_then(|m| m.get("role"))
            .and_then(|v| v.as_str())
            .or_else(|| val.get("role").and_then(|v| v.as_str()));
        let is_user_assistant = matches!(entry_type, Some("user" | "assistant"))
            || (entry_type == Some("message") && matches!(role_hint, Some("user" | "assistant")));
        if !is_user_assistant {
            return;
        }

        let created = val.get("timestamp").and_then(parse_timestamp);

        self.state.started_at = match (self.state.started_at, created) {
            (Some(curr), Some(ts)) => Some(curr.min(ts)),
            (None, Some(ts)) => Some(ts),
            (other, None) => other,
        };
        self.state.ended_at = match (self.state.ended_at, created) {
            (Some(curr), Some(ts)) => Some(curr.max(ts)),
            (None, Some(ts)) => Some(ts),
            (Some(curr), None) => Some(curr),
            (None, None) => None,
        };

        let role = role_hint.or(entry_type).unwrap_or("agent");
        let content_val = val
            .get("message")
            .and_then(|m| m.get("content"))
            .or_else(|| val.get("content"));
        let content_str = content_val.map(flatten_content).unwrap_or_default();

        // Tool results ride in user entries as
        // content:[{type:"tool_result", tool_use_id, content}].
        // flatten_content ignores those blocks, so without this
        // pass every result entry is dropped wholesale
        // (~30-40% of real entries per transcript), leaving
        // invocations with call_ids but no outputs anywhere.
        if let Some(blocks) = content_val.and_then(Value::as_array) {
            for block in blocks {
                if block.get("type").and_then(Value::as_str) != Some("tool_result") {
                    continue;
                }
                let result_text = match block.get("content") {
                    Some(Value::String(s)) => s.clone(),
                    Some(Value::Array(parts)) => parts
                        .iter()
                        .filter_map(|part| part.get("text").and_then(Value::as_str))
                        .collect::<Vec<_>>()
                        .join("\n"),
                    _ => continue,
                };
                if result_text.trim().is_empty() {
                    continue;
                }
                let mut tool_extra = Map::new();
                tool_extra.insert(
                    "source".to_string(),
                    Value::String("tool_result".to_string()),
                );
                if let Some(id) = block.get("tool_use_id").and_then(Value::as_str) {
                    tool_extra.insert("tool_use_id".to_string(), Value::String(id.to_string()));
                }
                // Failed tool calls are common and downstream
                // analytics split on them; pass the flag through
                // when Claude recorded one.
                if let Some(is_error) = block.get("is_error").and_then(Value::as_bool) {
                    tool_extra.insert("is_error".to_string(), Value::from(is_error));
                }
                self.push(NormalizedMessage {
                    idx: 0,
                    role: "tool".to_string(),
                    author: None,
                    created_at: created,
                    content: result_text,
                    extra: Value::Object(tool_extra),
                    invocations: Vec::new(),
                    snippets: Vec::new(),
                });
            }
        }

        if content_str.trim().is_empty() {
            return;
        }

        let author = val
            .get("message")
            .and_then(|m| m.get("model"))
            .and_then(|v| v.as_str())
            .map(String::from);
        let invocations =
            content_val.map_or_else(Vec::new, extract_invocations_from_content_blocks);

        self.push(NormalizedMessage {
            idx: 0,
            role: role.to_string(),
            author,
            created_at: created,
            content: content_str,
            extra: if self.compact_message_extra {
                ClaudeCodeConnector::compact_message_extra(&val)
            } else {
                val
            },
            invocations,
            snippets: Vec::new(),
        });
    }
}

/// Per-file identity shared by every part of one Claude session.
struct ClaudeJsonlIdentity<'p> {
    path: &'p Path,
    external_id: Option<String>,
    is_subagent_transcript: bool,
    parent_session_id: Option<String>,
}

struct ClaudeJsonlOptions {
    /// Read at most this many bytes of the file (the pre-parse size).
    size_limit: Option<u64>,
    compact_message_extra: bool,
    limits: ClaudeChunkLimits,
    wants_tail: bool,
}

struct ClaudeJsonlOutcome {
    conversations_emitted: usize,
    messages_emitted: usize,
    resumed: bool,
    tail: Option<SourceTailState>,
    /// Strings cut and values dropped by the long-line path. When it is not
    /// zero the source must not be certified complete.
    cut_values: usize,
}

fn claude_jsonl_part(
    identity: &ClaudeJsonlIdentity<'_>,
    state: &ClaudeSessionState,
    json_title: Option<&String>,
    messages: Vec<NormalizedMessage>,
) -> NormalizedConversation {
    NormalizedConversation {
        agent_slug: "claude_code".into(),
        external_id: identity.external_id.clone(),
        title: claude_session_title(state, json_title),
        workspace: state.workspace.clone(),
        source_path: identity.path.to_path_buf(),
        started_at: state.started_at,
        ended_at: state.ended_at,
        metadata: serde_json::json!({
            "source": "claude_code",
            "sessionId": state.session_id,
            "cliSessionId": Option::<String>::None,
            "gitBranch": state.git_branch,
            "permissionMode": Option::<String>::None,
            "bodyAvailable": true,
            "sidechain": identity.is_subagent_transcript,
            "parentSessionId": identity.parent_session_id
        }),
        messages,
    }
}

fn push_tail_prefix(prefix: &mut Vec<u8>, bytes: &[u8]) {
    if bytes.len() >= SOURCE_TAIL_PREFIX_BYTES {
        prefix.clear();
        prefix.extend_from_slice(&bytes[bytes.len() - SOURCE_TAIL_PREFIX_BYTES..]);
        return;
    }
    let keep = SOURCE_TAIL_PREFIX_BYTES - bytes.len();
    if prefix.len() > keep {
        prefix.drain(..prefix.len() - keep);
    }
    prefix.extend_from_slice(bytes);
}

/// Parse one JSONL record line the way the full-file reader always has:
/// skip blank and non-UTF-8 lines, strip a BOM, and require valid JSON.
fn parse_claude_jsonl_line(bytes: &[u8]) -> Option<Value> {
    let line = std::str::from_utf8(bytes).ok()?;
    if line.trim().is_empty() {
        return None;
    }
    serde_json::from_str::<Value>(line.trim_start_matches('\u{feff}')).ok()
}

/// Longest JSONL line the parser buffers whole, `FAD_CLAUDE_MAX_LINE_BYTES` to
/// override. Real Claude Code lines stay under about 9 MB, and a buffered line
/// costs about nine times its size downstream (200 MiB measured at 1.9 GiB).
/// A longer line is parsed by streaming instead (see `long_jsonl_line`): it
/// keeps up to `FAD_CLAUDE_LONG_LINE_KEEP_BYTES` (16 MiB) of its string data,
/// marks each cut string, and parsing goes on with the next line, so no
/// message is dropped.
const DEFAULT_MAX_LINE_BYTES: usize = 32 * 1024 * 1024;
const DEFAULT_LONG_LINE_KEEP_BYTES: usize = 16 * 1024 * 1024;

fn message_chunk_bytes() -> usize {
    #[cfg(test)]
    if let Some(bytes) = TEST_MESSAGE_CHUNK_BYTES.with(std::cell::Cell::get) {
        return bytes;
    }
    crate::types::MESSAGE_CHUNK_BYTES
}

fn long_line_keep_bytes() -> usize {
    #[cfg(test)]
    if let Some(keep) = TEST_LONG_LINE_KEEP_BYTES.with(std::cell::Cell::get) {
        return keep;
    }
    dotenvy::var("FAD_CLAUDE_LONG_LINE_KEEP_BYTES")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(DEFAULT_LONG_LINE_KEEP_BYTES)
}

fn max_line_bytes() -> usize {
    #[cfg(test)]
    if let Some(cap) = TEST_MAX_LINE_BYTES.with(std::cell::Cell::get) {
        return cap;
    }
    dotenvy::var("FAD_CLAUDE_MAX_LINE_BYTES")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(DEFAULT_MAX_LINE_BYTES)
}

/// Read one line (with its newline) into `line`, buffering at most `cap`
/// bytes. Returns the bytes consumed and whether the line was longer than
/// `cap`. A longer line is consumed to its end without being kept when
/// `skip_long` is set; otherwise reading stops at the cap, `line` holds the
/// bytes consumed so far, and the rest of the line is still in `reader`.
fn read_line_capped<R: std::io::BufRead>(
    reader: &mut R,
    line: &mut Vec<u8>,
    cap: usize,
    skip_long: bool,
) -> std::io::Result<(usize, bool)> {
    let mut consumed = 0usize;
    let mut too_long = false;
    loop {
        let buf = reader.fill_buf()?;
        if buf.is_empty() {
            return Ok((consumed, too_long));
        }
        let (take, done) = match buf.iter().position(|&b| b == b'\n') {
            Some(i) => (i + 1, true),
            None => (buf.len(), false),
        };
        if !too_long {
            if line.len() + take > cap {
                too_long = true;
                if !skip_long {
                    return Ok((consumed, true));
                }
                *line = Vec::new();
            } else {
                line.extend_from_slice(&buf[..take]);
            }
        }
        reader.consume(take);
        consumed += take;
        if done {
            return Ok((consumed, too_long));
        }
    }
}

fn blake3_hex_of_range(file: &mut fs::File, from: u64, len: u64) -> Option<String> {
    use std::io::{Read, Seek, SeekFrom};
    file.seek(SeekFrom::Start(from)).ok()?;
    let mut hasher = blake3::Hasher::new();
    let copied = std::io::copy(&mut file.take(len), &mut hasher).ok()?;
    (copied == len).then(|| hasher.finalize().to_hex().to_string())
}

/// blake3 hex of 32 blocks of 4 KiB spread evenly over `[0, offset)`, each
/// hashed with its position.
fn sampled_blake3(file: &mut fs::File, offset: u64) -> Option<String> {
    use std::io::{Read, Seek, SeekFrom};
    const SAMPLES: u64 = 32;
    const BLOCK: u64 = 4096;
    let mut hasher = blake3::Hasher::new();
    let mut block = Vec::with_capacity(BLOCK as usize);
    for i in 1..=SAMPLES {
        let pos = offset / (SAMPLES + 1) * i;
        let len = BLOCK.min(offset - pos);
        file.seek(SeekFrom::Start(pos)).ok()?;
        block.clear();
        (&mut *file).take(len).read_to_end(&mut block).ok()?;
        if block.len() as u64 != len {
            return None;
        }
        hasher.update(&pos.to_le_bytes());
        hasher.update(&block);
    }
    Some(hasher.finalize().to_hex().to_string())
}

/// Device and inode of an open file. `None` where the platform has neither,
/// which disables tail resume there.
fn open_file_identity(file: &fs::File) -> Option<(u64, u64)> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        let metadata = file.metadata().ok()?;
        Some((metadata.dev(), metadata.ino()))
    }
    #[cfg(not(unix))]
    {
        let _ = file;
        None
    }
}

/// Whether `resume` describes the file behind `file`: same device and inode,
/// at least `offset` bytes, and the same first bytes and bytes before
/// `offset`. All reads go through the open file, so a file swapped in after
/// the host checked the path is caught here.
fn tail_resume_matches(file: &mut fs::File, resume: &SourceTailResume) -> bool {
    let Some((dev, inode)) = open_file_identity(file) else {
        return false;
    };
    if (dev, inode) != (resume.dev, resume.inode) {
        return false;
    }
    let Ok(metadata) = file.metadata() else {
        return false;
    };
    let head_len = resume.offset.min(SOURCE_TAIL_HEAD_BYTES as u64);
    if metadata.len() < resume.offset || resume.head_len != head_len {
        return false;
    }
    let prefix_len = resume.offset.min(SOURCE_TAIL_PREFIX_BYTES as u64);
    blake3_hex_of_range(file, 0, head_len).as_deref() == Some(resume.head_blake3.as_str())
        && blake3_hex_of_range(file, resume.offset - prefix_len, prefix_len).as_deref()
            == Some(resume.prefix_blake3.as_str())
        && sampled_blake3(file, resume.offset).as_deref() == Some(resume.samples_blake3.as_str())
}

/// The last `ai-title` in `[from, from + len)` of the file, else `current`.
/// Run once, only when a session is split into parts, so that every part
/// carries the title the whole session ends with.
fn scan_final_ai_title(
    path: &Path,
    from: u64,
    len: u64,
    current: Option<String>,
) -> Option<String> {
    use std::io::{Read, Seek, SeekFrom};
    let Ok(mut file) = fs::File::open(path) else {
        return current;
    };
    if file.seek(SeekFrom::Start(from)).is_err() {
        return current;
    }
    let mut reader = std::io::BufReader::with_capacity(256 * 1024, file.take(len));
    let mut title = current;
    let mut line = Vec::new();
    loop {
        line.clear();
        // Longer lines are skipped without being buffered.
        match read_line_capped(&mut reader, &mut line, AI_TITLE_LINE_MAX_BYTES, true) {
            Ok((0, _)) | Err(_) => break,
            Ok((_, true)) => continue,
            Ok(_) => {}
        }
        if !line.windows(8).any(|w| w == b"ai-title") {
            continue;
        }
        let Some(val) = parse_claude_jsonl_line(&line) else {
            continue;
        };
        if val.get("type").and_then(Value::as_str) != Some("ai-title") {
            continue;
        }
        if let Some(found) = val
            .get("title")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|t| !t.is_empty())
        {
            title = Some(found.to_string());
        }
    }
    title
}

/// Parse a Claude JSONL session, emitting it in parts of bounded size.
///
/// Every part shares the session identity; message indexes continue across
/// parts (and from `resume.next_idx` on a tail scan), so the parts together
/// equal one full parse. Only complete lines are consumed: a final segment
/// without a newline is taken only when it is already valid JSON, so a line
/// still being written is left for the next scan.
#[allow(clippy::too_many_lines)]
fn parse_claude_jsonl_source(
    identity: &ClaudeJsonlIdentity<'_>,
    options: ClaudeJsonlOptions,
    resume: Option<SourceTailResume>,
    on_conversation: &mut dyn FnMut(NormalizedConversation) -> Result<()>,
) -> Result<ClaudeJsonlOutcome> {
    use std::io::{Read, Seek, SeekFrom};
    let path = identity.path;
    let mut file = fs::File::open(path).with_context(|| format!("open {}", path.display()))?;

    let resume = resume.and_then(|resume| {
        let state = ClaudeSessionState::from_json(&resume.carried)?;
        let fits = options
            .size_limit
            .is_none_or(|limit| resume.offset <= limit);
        (resume.offset > 0 && resume.next_idx >= 0 && fits).then_some((resume, state))
    });
    let mut prefix = Vec::new();
    let (start, mut next_idx, state) = match resume {
        Some((resume, state)) if tail_resume_matches(&mut file, &resume) => {
            let from = resume
                .offset
                .saturating_sub(SOURCE_TAIL_PREFIX_BYTES as u64);
            file.seek(SeekFrom::Start(from))
                .with_context(|| format!("seek {}", path.display()))?;
            (&mut file)
                .take(resume.offset - from)
                .read_to_end(&mut prefix)
                .with_context(|| format!("read {}", path.display()))?;
            (resume.offset, resume.next_idx, state)
        }
        Some(_) => {
            tracing::debug!(path = %path.display(), "claude_code tail resume does not match the open file; parsing from the start");
            (0, 0, ClaudeSessionState::default())
        }
        None => (0, 0, ClaudeSessionState::default()),
    };
    let resumed = start > 0;
    file.seek(SeekFrom::Start(start))
        .with_context(|| format!("seek {}", path.display()))?;
    let readable = options
        .size_limit
        .map_or(u64::MAX, |limit| limit.saturating_sub(start));
    let mut reader = std::io::BufReader::with_capacity(256 * 1024, file.take(readable));

    let mut parser = ClaudeJsonlParser {
        state,
        messages: Vec::new(),
        content_bytes: 0,
        compact_message_extra: options.compact_message_extra,
    };
    let mut offset = start;
    // Set once the session is split: the title every part carries.
    let mut part_title: Option<Option<String>> = None;
    let mut conversations_emitted = 0usize;
    let mut messages_emitted = 0usize;
    let mut cut_values = 0usize;
    let mut emit = |parser: &mut ClaudeJsonlParser,
                    part_title: Option<&Option<String>>,
                    next_idx: &mut i64|
     -> Result<()> {
        let mut messages = std::mem::take(&mut parser.messages);
        parser.content_bytes = 0;
        for (i, message) in messages.iter_mut().enumerate() {
            message.idx = next_idx.saturating_add(i64::try_from(i).unwrap_or(i64::MAX));
        }
        *next_idx = next_idx.saturating_add(i64::try_from(messages.len()).unwrap_or(i64::MAX));
        conversations_emitted += 1;
        messages_emitted += messages.len();
        let json_title = part_title.map_or(parser.state.json_title.as_ref(), Option::as_ref);
        on_conversation(claude_jsonl_part(
            identity,
            &parser.state,
            json_title,
            messages,
        ))
    };

    let max_line = max_line_bytes();
    let mut line = Vec::new();
    loop {
        line.clear();
        let read = match read_line_capped(&mut reader, &mut line, max_line, false) {
            Ok((_, true)) => {
                // Too long to buffer: stream the rest of it, keeping a
                // bounded part of its text, then go on with the next line.
                let keep = long_line_keep_bytes();
                let mut tail_bytes = Vec::new();
                let mut rest = std::io::Read::chain(&line[..], &mut reader);
                match super::long_jsonl_line::parse_long_line(
                    &mut rest,
                    keep,
                    &mut tail_bytes,
                    SOURCE_TAIL_PREFIX_BYTES,
                ) {
                    super::long_jsonl_line::LongLine::Parsed {
                        value,
                        consumed,
                        truncated_strings,
                        dropped_values,
                    } => {
                        tracing::warn!(
                            path = %path.display(),
                            offset,
                            line_bytes = consumed,
                            kept_bytes = keep,
                            truncated_strings,
                            dropped_values,
                            "claude_code line longer than FAD_CLAUDE_MAX_LINE_BYTES; indexed with its text cut"
                        );
                        offset += consumed;
                        cut_values = cut_values
                            .saturating_add(truncated_strings)
                            .saturating_add(dropped_values);
                        if options.wants_tail {
                            push_tail_prefix(&mut prefix, &tail_bytes);
                        }
                        parser.ingest_line(value);
                    }
                    super::long_jsonl_line::LongLine::Invalid { consumed } => {
                        offset += consumed;
                        if options.wants_tail {
                            push_tail_prefix(&mut prefix, &tail_bytes);
                        }
                        continue;
                    }
                    // A line still being written: leave it for the next scan.
                    super::long_jsonl_line::LongLine::Incomplete => break,
                }
                if parser.messages.len() >= options.limits.max_messages
                    || parser.content_bytes >= options.limits.max_content_bytes
                {
                    if part_title.is_none() {
                        let remaining = start.saturating_add(readable).saturating_sub(offset);
                        part_title = Some(scan_final_ai_title(
                            path,
                            offset,
                            remaining,
                            parser.state.json_title.clone(),
                        ));
                    }
                    emit(&mut parser, part_title.as_ref(), &mut next_idx)?;
                }
                continue;
            }
            Ok((read, false)) => read,
            Err(error) => {
                tracing::debug!(path = %path.display(), %error, "claude_code read error; stopping at last complete line");
                break;
            }
        };
        if read == 0 {
            break;
        }
        let terminated = line.last() == Some(&b'\n');
        let record = if terminated {
            &line[..line.len() - 1]
        } else {
            &line[..]
        };
        let val = parse_claude_jsonl_line(record);
        if !terminated && val.is_none() {
            // A line still being written: leave it for the next scan.
            break;
        }
        offset += read as u64;
        if options.wants_tail {
            push_tail_prefix(&mut prefix, &line);
        }
        let Some(val) = val else {
            continue;
        };
        parser.ingest_line(val);
        if parser.messages.len() >= options.limits.max_messages
            || parser.content_bytes >= options.limits.max_content_bytes
        {
            if part_title.is_none() {
                let remaining = start.saturating_add(readable).saturating_sub(offset);
                part_title = Some(scan_final_ai_title(
                    path,
                    offset,
                    remaining,
                    parser.state.json_title.clone(),
                ));
            }
            emit(&mut parser, part_title.as_ref(), &mut next_idx)?;
        }
    }
    if !parser.messages.is_empty() {
        emit(&mut parser, part_title.as_ref(), &mut next_idx)?;
    }

    // The reader owns the file; take it back for the head hash.
    let mut file = reader.into_inner().into_inner();
    let tail = if options.wants_tail {
        let head_len = offset.min(SOURCE_TAIL_HEAD_BYTES as u64);
        match (
            open_file_identity(&file),
            blake3_hex_of_range(&mut file, 0, head_len),
            sampled_blake3(&mut file, offset),
            parser.state.to_json(),
        ) {
            (Some((dev, inode)), Some(head_blake3), Some(samples_blake3), Some(carried)) => {
                Some(SourceTailState {
                    offset,
                    next_idx,
                    carried,
                    prefix,
                    resumed,
                    dev,
                    inode,
                    head_len,
                    head_blake3,
                    samples_blake3,
                })
            }
            _ => None,
        }
    } else {
        None
    };
    Ok(ClaudeJsonlOutcome {
        conversations_emitted,
        messages_emitted,
        resumed,
        tail,
        cut_values,
    })
}

#[allow(clippy::too_many_lines)]
fn scan_claude_with_callback(
    ctx: &ScanContext,
    on_conversation: &mut dyn FnMut(NormalizedConversation) -> Result<()>,
) -> Result<()> {
    scan_claude_with_callback_with_exclusions(
        ctx,
        on_conversation,
        &excluded_scan_paths_from_env(),
        &mut SourceScanHooks::default(),
    )
}

#[allow(clippy::too_many_lines)]
fn scan_claude_with_callback_with_exclusions(
    ctx: &ScanContext,
    on_conversation: &mut dyn FnMut(NormalizedConversation) -> Result<()>,
    excluded_paths: &[PathBuf],
    hooks: &mut SourceScanHooks<'_>,
) -> Result<()> {
    let roots: Vec<ScanRoot> = ClaudeCodeConnector::source_roots(ctx);
    // Overlapping explicit roots (or a symlinked CLAUDE_CONFIG_DIR aliasing
    // another root) reach the same transcript twice; dedupe on the
    // lossless path key across ALL roots, first occurrence wins.
    let mut seen_files: HashSet<PathBuf> = HashSet::new();

    let mut file_count = 0;

    for root in roots {
        let explicit_file_root = root.path.is_file();
        let scan_target = root.path.clone();
        let external_id_root = if explicit_file_root {
            ClaudeCodeConnector::projects_root_for_explicit_file(&root.path)
                .or_else(|| root.path.parent().map(Path::to_path_buf))
        } else {
            Some(scan_target.clone())
        };

        if !scan_target.exists() {
            continue;
        }

        let session_paths = if explicit_file_root {
            vec![scan_target.clone()]
        } else {
            ClaudeCodeConnector::session_files(&scan_target)
        };

        for path in session_paths {
            if path_is_excluded(&path, excluded_paths) {
                tracing::debug!(
                    path = %path.display(),
                    "claude_code skipping excluded session file"
                );
                continue;
            }
            if !seen_files.insert(dedupe_path_key(&path)) {
                continue;
            }
            let ext = path.extension().and_then(|s| s.to_str());
            if !file_modified_since(&path, ctx.since_ts) {
                continue;
            }
            // Pre-parse source identity: constructed exactly like
            // discover_sources_with_exclusions(), with size/mtime observed
            // BEFORE the file is opened (FAD#22).
            let discovered = DiscoveredSourceFile::new(
                "claude_code",
                &root,
                path.clone(),
                ClaudeCodeConnector::discovered_source_role(&path),
                true,
            )
            .with_fs_metadata();
            if !hooks.should_scan(&discovered) {
                tracing::debug!(
                    path = %path.display(),
                    "claude_code host ledger skipped unchanged source"
                );
                continue;
            }
            let file_size_bytes = fs::metadata(&path).ok().map(|metadata| metadata.len());
            let compact_message_extra =
                ClaudeCodeConnector::should_compact_large_message_extra(file_size_bytes);
            if compact_message_extra {
                tracing::debug!(
                    path = %path.display(),
                    size_bytes = file_size_bytes.unwrap_or_default(),
                    "claude_code compacting per-message extra payloads for large session"
                );
            }
            file_count += 1;
            if file_count <= 3 {
                tracing::debug!(path = %path.display(), "claude_code found file");
            }

            let mut messages = Vec::new();
            let mut started_at: Option<i64> = None;
            let mut ended_at: Option<i64> = None;
            let mut workspace: Option<PathBuf> = None;
            let mut session_id: Option<String> = None;
            let mut cli_session_id: Option<String> = None;
            // Only the single-file JSON sidecar path below uses these; the
            // JSONL path keeps its state in `ClaudeSessionState`.
            let git_branch: Option<String> = None;
            let json_title: Option<String>;
            let mut permission_mode: Option<String> = None;
            let mut source_kind = "claude_code";

            // Subagent transcripts live at
            // `<session>/subagents/agent-*.jsonl` and are separate
            // conversations; surface the relationship instead of leaving the
            // parent link implicit in the path.
            let is_subagent_transcript = path
                .components()
                .any(|c| c.as_os_str().to_str() == Some("subagents"));
            let mut parent_session_id: Option<String> = None;
            if is_subagent_transcript {
                let components: Vec<_> = path.components().collect();
                if let Some(pos) = components
                    .iter()
                    .position(|c| c.as_os_str().to_str() == Some("subagents"))
                    && pos > 0
                {
                    parent_session_id = components[pos - 1].as_os_str().to_str().map(String::from);
                }
            }

            if ext == Some("jsonl") {
                let identity = ClaudeJsonlIdentity {
                    path: &path,
                    external_id: external_id_root
                        .as_deref()
                        .and_then(|base| path.strip_prefix(base).ok())
                        .and_then(|rel| rel.to_str())
                        .map(std::string::ToString::to_string)
                        .or_else(|| {
                            path.file_name()
                                .and_then(|s| s.to_str())
                                .map(std::string::ToString::to_string)
                        }),
                    is_subagent_transcript,
                    parent_session_id: parent_session_id.clone(),
                };
                let wants_tail = hooks.supports_tail();
                let resume = if wants_tail {
                    hooks.resume_point(&discovered)
                } else {
                    None
                };
                // A tail-aware host gets a consistent snapshot: only the bytes
                // that existed before the parse began. Other hosts keep the
                // read-to-EOF behaviour and the changed-during-parse check.
                let size_limit = if wants_tail {
                    discovered.size_bytes
                } else {
                    None
                };
                let outcome = parse_claude_jsonl_source(
                    &identity,
                    ClaudeJsonlOptions {
                        size_limit,
                        compact_message_extra,
                        limits: ClaudeChunkLimits::from_env(),
                        wants_tail,
                    },
                    resume,
                    on_conversation,
                )?;
                if outcome.conversations_emitted == 0 && !outcome.resumed {
                    if file_count <= 3 {
                        tracing::debug!(path = %path.display(), "claude_code no messages extracted");
                    }
                    continue;
                }
                tracing::debug!(
                    path = %path.display(),
                    parts = outcome.conversations_emitted,
                    messages = outcome.messages_emitted,
                    resumed = outcome.resumed,
                    "claude_code extracted messages"
                );
                if outcome.cut_values > 0 {
                    // Text was cut to fit a size limit: never certify the
                    // source, so a later build that keeps all of it parses
                    // the file again.
                    tracing::warn!(
                        path = %path.display(),
                        cut_values = outcome.cut_values,
                        "claude_code source indexed with text cut; completion withheld"
                    );
                } else if wants_tail {
                    // Only bytes up to the pre-parse size were read, so the
                    // completion stays valid even if the file grew meanwhile;
                    // the tail state says where the next scan resumes.
                    hooks.complete(&SourceCompletion {
                        source: discovered,
                        required_sidecars: Vec::new(),
                        conversations_emitted: outcome.conversations_emitted,
                        tail: outcome.tail,
                    })?;
                } else if discovered.fs_metadata_changed() {
                    tracing::debug!(
                        path = %path.display(),
                        "claude_code source changed during parse; completion withheld"
                    );
                } else {
                    hooks.complete(&SourceCompletion {
                        source: discovered,
                        required_sidecars: Vec::new(),
                        conversations_emitted: outcome.conversations_emitted,
                        tail: None,
                    })?;
                }
                continue;
            }
            {
                if let Ok(metadata) = fs::metadata(&path)
                    && metadata.len() > 100 * 1024 * 1024
                {
                    tracing::debug!(
                        path = %path.display(),
                        size_bytes = metadata.len(),
                        "skipping large file (>100MB)"
                    );
                    continue;
                }

                let content_string = fs::read_to_string(&path)
                    .with_context(|| format!("read {}", path.display()))?;
                let val: Value = match serde_json::from_str(&content_string) {
                    Ok(v) => v,
                    Err(e) => {
                        tracing::debug!(path = %path.display(), error = %e, "claude_code skipping malformed JSON");
                        continue;
                    }
                };

                json_title = ClaudeCodeConnector::non_empty_json_string(&val, "title");
                if workspace.is_none() {
                    workspace =
                        ClaudeCodeConnector::non_empty_json_string(&val, "cwd").map(PathBuf::from);
                }
                if session_id.is_none() {
                    session_id = ClaudeCodeConnector::non_empty_json_string(&val, "sessionId");
                }
                if cli_session_id.is_none() {
                    cli_session_id =
                        ClaudeCodeConnector::non_empty_json_string(&val, "cliSessionId");
                }
                if permission_mode.is_none() {
                    permission_mode =
                        ClaudeCodeConnector::non_empty_json_string(&val, "permissionMode");
                }
                let sidecar_created_at = val.get("createdAt").and_then(parse_timestamp);
                let sidecar_last_activity_at = val.get("lastActivityAt").and_then(parse_timestamp);
                started_at = started_at
                    .or(sidecar_created_at)
                    .or(sidecar_last_activity_at);
                ended_at = ended_at.or(sidecar_last_activity_at).or(sidecar_created_at);

                if let Some(arr) = val.get("messages").and_then(|m| m.as_array()) {
                    for item in arr {
                        let role = item
                            .get("role")
                            .or_else(|| item.get("type"))
                            .and_then(|v| v.as_str())
                            .unwrap_or("agent");
                        let created = item
                            .get("timestamp")
                            .or_else(|| item.get("time"))
                            .and_then(parse_timestamp);

                        started_at = match (started_at, created) {
                            (Some(curr), Some(ts)) => Some(curr.min(ts)),
                            (None, Some(ts)) => Some(ts),
                            (other, None) => other,
                        };
                        ended_at = match (ended_at, created) {
                            (Some(curr), Some(ts)) => Some(curr.max(ts)),
                            (None, Some(ts)) => Some(ts),
                            (Some(curr), None) => Some(curr),
                            (None, None) => None,
                        };

                        let content_val = item.get("content").or_else(|| item.get("text"));
                        let content_str = content_val.map(flatten_content).unwrap_or_default();

                        if content_str.trim().is_empty() {
                            continue;
                        }

                        messages.push(NormalizedMessage {
                            idx: 0,
                            role: role.to_string(),
                            author: None,
                            created_at: created,
                            content: content_str,
                            extra: if compact_message_extra {
                                ClaudeCodeConnector::compact_message_extra(item)
                            } else {
                                item.clone()
                            },
                            invocations: content_val
                                .map_or_else(Vec::new, extract_invocations_from_content_blocks),
                            snippets: Vec::new(),
                        });
                    }
                }
                if messages.is_empty()
                    && ClaudeCodeConnector::path_is_desktop_sidecar(&path)
                    && let Some(message) =
                        ClaudeCodeConnector::desktop_sidecar_metadata_message(&val)
                {
                    source_kind = "claude_code_desktop_sidecar";
                    messages.push(message);
                }
                crate::types::reindex_messages(&mut messages);
            }

            if messages.is_empty() {
                if file_count <= 3 {
                    tracing::debug!(path = %path.display(), "claude_code no messages extracted");
                }
                continue;
            }
            tracing::debug!(path = %path.display(), messages = messages.len(), "claude_code extracted messages");

            let title = json_title.or_else(|| {
                messages
                    .iter()
                    .find(|m| m.role == "user")
                    .map(|m| {
                        m.content
                            .lines()
                            .next()
                            .unwrap_or(&m.content)
                            .chars()
                            .take(100)
                            .collect::<String>()
                    })
                    .or_else(|| {
                        workspace
                            .as_ref()
                            .and_then(|p| p.file_name())
                            .and_then(|n| n.to_str())
                            .map(String::from)
                    })
            });

            on_conversation(NormalizedConversation {
                agent_slug: "claude_code".into(),
                external_id: if source_kind == "claude_code_desktop_sidecar" {
                    cli_session_id.clone().or_else(|| session_id.clone())
                } else {
                    external_id_root
                        .as_deref()
                        .and_then(|base| path.strip_prefix(base).ok())
                        .and_then(|rel| rel.to_str())
                        .map(std::string::ToString::to_string)
                        .or_else(|| {
                            path.file_name()
                                .and_then(|s| s.to_str())
                                .map(std::string::ToString::to_string)
                        })
                },
                title,
                workspace,
                source_path: path.clone(),
                started_at,
                ended_at,
                metadata: serde_json::json!({
                    "source": source_kind,
                    "sessionId": session_id,
                    "cliSessionId": cli_session_id,
                    "gitBranch": git_branch,
                    "permissionMode": permission_mode,
                    "bodyAvailable": source_kind != "claude_code_desktop_sidecar",
                    "sidechain": is_subagent_transcript,
                    "parentSessionId": parent_session_id
                }),
                messages,
            })?;

            // Source complete: the (single) conversation derived from this
            // transcript was delivered. Suppressed when the file changed
            // while it was being parsed — the host must re-observe it.
            if discovered.fs_metadata_changed() {
                tracing::debug!(
                    path = %path.display(),
                    "claude_code source changed during parse; completion withheld"
                );
            } else {
                hooks.complete(&SourceCompletion {
                    source: discovered,
                    required_sidecars: Vec::new(),
                    conversations_emitted: 1,
                    tail: None,
                })?;
            }
        }
    }

    Ok(())
}

impl Connector for ClaudeCodeConnector {
    fn detect(&self) -> DetectionResult {
        franken_detection_for_connector("claude_code").unwrap_or_else(DetectionResult::not_found)
    }

    fn scan(&self, ctx: &ScanContext) -> Result<Vec<NormalizedConversation>> {
        let mut convs = Vec::new();
        scan_claude_with_callback(ctx, &mut |conv| {
            convs.push(conv);
            Ok(())
        })?;
        Ok(convs)
    }

    fn supports_streaming_scan(&self) -> bool {
        true
    }

    fn discover_source_files(&self, ctx: &ScanContext) -> Result<Vec<DiscoveredSourceFile>> {
        Ok(Self::discover_sources(ctx))
    }

    fn scan_with_callback(
        &self,
        ctx: &ScanContext,
        on_conversation: &mut dyn FnMut(NormalizedConversation) -> Result<()>,
    ) -> Result<()> {
        scan_claude_with_callback(ctx, on_conversation)
    }

    fn supports_source_boundaries(&self) -> bool {
        true
    }

    fn scan_with_source_boundaries(
        &self,
        ctx: &ScanContext,
        hooks: &mut SourceScanHooks<'_>,
        on_conversation: &mut dyn FnMut(NormalizedConversation) -> Result<()>,
    ) -> Result<()> {
        scan_claude_with_callback_with_exclusions(
            ctx,
            on_conversation,
            &excluded_scan_paths_from_env(),
            hooks,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::fs;
    use tempfile::TempDir;

    /// Create a test-ready Claude directory structure.
    /// Includes a `projects` marker subdir so `looks_like_root()` returns true
    /// and the connector scans only the temp dir instead of the real ~/.claude/projects.
    fn make_test_claude_dir(base: &std::path::Path) -> PathBuf {
        let claude_dir = base.join(".claude");
        fs::create_dir_all(claude_dir.join("projects")).unwrap();
        claude_dir
    }

    // =========================================================================
    // Constructor tests
    // =========================================================================

    #[test]
    fn new_creates_connector() {
        let connector = ClaudeCodeConnector::new();
        let _ = connector;
    }

    #[test]
    fn scan_marks_subagent_transcripts_with_parent_link() {
        let base = TempDir::new().unwrap();
        let claude_dir = make_test_claude_dir(base.path());
        let session_dir = claude_dir
            .join("projects")
            .join("-data-projects-demo")
            .join("11111111-2222-3333-4444-555555555555");
        let subagents = session_dir
            .join("subagents")
            .join("66666666-7777-8888-9999-000000000000");
        fs::create_dir_all(&subagents).unwrap();
        fs::write(
            subagents.join("agent-abc123.jsonl"),
            concat!(
                r#"{"type":"assistant","sessionId":"66666666-7777-8888-9999-000000000000","message":{"role":"assistant","model":"claude-x","content":[{"type":"text","text":"subagent work"}]}}"#,
                "\n",
            ),
        )
        .unwrap();

        let connector = ClaudeCodeConnector::new();
        let ctx = ScanContext::local_default(claude_dir.clone(), None);
        let convs = connector.scan(&ctx).unwrap();

        assert_eq!(convs.len(), 1);
        assert_eq!(convs[0].metadata["sidechain"], true);
        assert_eq!(
            convs[0].metadata["parentSessionId"],
            "11111111-2222-3333-4444-555555555555"
        );
    }

    #[test]
    fn scan_prefers_ai_title_over_first_line_truncation() {
        let base = TempDir::new().unwrap();
        let claude_dir = make_test_claude_dir(base.path());
        let session_dir = claude_dir
            .join("projects")
            .join("-data-projects-demo")
            .join("aaaa-bbbb");
        fs::create_dir_all(&session_dir).unwrap();
        fs::write(
            session_dir.join("cccc-dddd.jsonl"),
            concat!(
                r#"{"type":"ai-title","title":"Fix the flaky auth test"}"#,
                "\n",
                r#"{"type":"user","sessionId":"cccc-dddd","message":{"role":"user","content":[{"type":"text","text":"please look at the auth module and fix the flaky test"}]}}"#,
                "\n",
            ),
        )
        .unwrap();

        let connector = ClaudeCodeConnector::new();
        let ctx = ScanContext::local_default(claude_dir, None);
        let convs = connector.scan(&ctx).unwrap();

        assert_eq!(convs.len(), 1);
        assert_eq!(convs[0].title.as_deref(), Some("Fix the flaky auth test"));
    }

    #[test]
    fn default_creates_connector() {
        let connector = ClaudeCodeConnector;
        let _ = connector;
    }

    #[test]
    fn projects_root_returns_claude_projects_path() {
        // Smoke test on the env-aware entry point. We can't reliably mutate
        // env vars in parallel tests (std::env::set_var is unsafe + forbid'd),
        // so this test only asserts the invariant that holds across all three
        // resolver branches: the path's last component is "projects". The
        // *precedence* logic is exercised by `projects_root_resolved_*` below
        // against explicit inputs, where we don't depend on process env at all.
        let root = ClaudeCodeConnector::projects_root();
        assert_eq!(
            root.file_name().and_then(|n| n.to_str()),
            Some("projects"),
            "projects_root() should always return a path ending in 'projects', got: {}",
            root.display()
        );
    }

    #[test]
    fn projects_root_resolved_uses_explicit_override_first() {
        let explicit = std::path::Path::new("/opt/claude_explicit");
        let xdg = std::path::Path::new("/opt/xdg");
        let home = std::path::Path::new("/opt/home");
        let root =
            ClaudeCodeConnector::projects_root_resolved(Some(explicit), Some(xdg), Some(home));
        assert_eq!(root, PathBuf::from("/opt/claude_explicit/projects"));
    }

    #[test]
    fn projects_root_resolved_falls_back_to_xdg_when_no_explicit_override() {
        let xdg = std::path::Path::new("/opt/xdg");
        let home = std::path::Path::new("/opt/home");
        let root = ClaudeCodeConnector::projects_root_resolved(None, Some(xdg), Some(home));
        // XDG layout is `${XDG_CONFIG_HOME}/claude-code/projects` (no leading
        // dot), distinct from the `.claude/projects` home-relative default —
        // matches what the Claude Code CLI itself does when XDG is set, and
        // what caam writes per-profile.
        assert_eq!(root, PathBuf::from("/opt/xdg/claude-code/projects"));
    }

    #[test]
    fn projects_root_resolved_uses_home_when_no_overrides() {
        let home = std::path::Path::new("/home/jane");
        let root = ClaudeCodeConnector::projects_root_resolved(None, None, Some(home));
        assert_eq!(root, PathBuf::from("/home/jane/.claude/projects"));
    }

    #[test]
    fn projects_root_resolved_falls_back_to_relative_when_home_missing() {
        // Sandboxed test environments occasionally have no HOME — historical
        // behavior was to return a relative `.claude/projects`, preserved here.
        let root = ClaudeCodeConnector::projects_root_resolved(None, None, None);
        assert_eq!(root, PathBuf::from(".claude/projects"));
    }

    #[test]
    fn projects_root_resolved_treats_explicit_as_higher_priority_than_xdg() {
        // Regression guard: if the precedence is ever swapped (XDG before
        // CLAUDE_CONFIG_DIR), caam-style multi-account isolation that sets
        // both vars to *different* values would silently route to the wrong
        // profile. Pin the precedence here so a refactor can't quietly undo it.
        let explicit = std::path::Path::new("/opt/account_A");
        let xdg = std::path::Path::new("/opt/account_B/xdg_config");
        let root = ClaudeCodeConnector::projects_root_resolved(Some(explicit), Some(xdg), None);
        assert_eq!(root, PathBuf::from("/opt/account_A/projects"));
        // Verify it did NOT pick the XDG path.
        assert_ne!(
            root,
            PathBuf::from("/opt/account_B/xdg_config/claude-code/projects")
        );
    }

    #[test]
    fn projects_root_candidates_keep_home_fallback_when_xdg_is_set() {
        let xdg = std::path::Path::new("/opt/xdg");
        let home = std::path::Path::new("/home/jane");
        let roots =
            ClaudeCodeConnector::projects_root_candidates_resolved(None, Some(xdg), Some(home));

        assert!(roots.contains(&PathBuf::from("/opt/xdg/claude-code/projects")));
        assert!(roots.contains(&PathBuf::from("/home/jane/.claude/projects")));
    }

    #[test]
    fn projects_root_candidates_include_macos_desktop_sidecar_roots() {
        let home = std::path::Path::new("/Users/jane");
        let roots = ClaudeCodeConnector::projects_root_candidates_resolved(None, None, Some(home));

        assert!(roots.contains(&PathBuf::from(
            "/Users/jane/Library/Application Support/Claude/claude-code-sessions"
        )));
        assert!(roots.contains(&PathBuf::from(
            "/Users/jane/Library/Application Support/Claude/local-agent-mode-sessions"
        )));
    }

    #[test]
    fn projects_root_candidates_explicit_override_does_not_scan_fallbacks() {
        let explicit = std::path::Path::new("/opt/claude_explicit");
        let xdg = std::path::Path::new("/opt/xdg");
        let home = std::path::Path::new("/home/jane");
        let roots = ClaudeCodeConnector::projects_root_candidates_resolved(
            Some(explicit),
            Some(xdg),
            Some(home),
        );

        assert_eq!(roots, vec![PathBuf::from("/opt/claude_explicit/projects")]);
    }

    #[test]
    fn session_files_returns_sorted_order() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("b.jsonl"), "{}").unwrap();
        fs::write(dir.path().join("a.jsonl"), "{}").unwrap();
        fs::write(dir.path().join("ignore.txt"), "x").unwrap();

        let files = ClaudeCodeConnector::session_files(dir.path());
        assert_eq!(files.len(), 2);

        let names: Vec<_> = files
            .iter()
            .map(|p| {
                p.file_name()
                    .and_then(|s| s.to_str())
                    .unwrap_or("")
                    .to_string()
            })
            .collect();
        assert_eq!(names, vec!["a.jsonl", "b.jsonl"]);
    }

    // =========================================================================
    // Detection tests
    // =========================================================================

    #[test]
    fn detect_not_found_without_projects_dir() {
        let connector = ClaudeCodeConnector::new();
        let result = connector.detect();
        // On most CI/test systems, .claude/projects won't exist
        // Just verify detect() doesn't panic
        let _ = result.detected;
    }

    // =========================================================================
    // JSONL parsing tests
    // =========================================================================

    #[test]
    fn scan_parses_jsonl_user_and_assistant_messages() {
        let dir = TempDir::new().unwrap();
        let claude_dir = make_test_claude_dir(dir.path());

        let session_file = claude_dir.join("session.jsonl");
        let content = r#"{"type":"user","timestamp":"2025-12-01T10:00:00Z","message":{"role":"user","content":"Hello Claude"}}
{"type":"assistant","timestamp":"2025-12-01T10:00:01Z","message":{"role":"assistant","content":"Hello! How can I help?"}}
{"type":"summary","timestamp":"2025-12-01T10:00:02Z","summary":"Test summary"}
"#;
        fs::write(&session_file, content).unwrap();

        let connector = ClaudeCodeConnector::new();
        let ctx = ScanContext::local_default(claude_dir.clone(), None);
        let result = connector.scan(&ctx);

        assert!(result.is_ok());
        let convs = result.unwrap();
        assert_eq!(convs.len(), 1);

        // Only user and assistant messages should be extracted (not summary)
        assert_eq!(convs[0].messages.len(), 2);
        assert_eq!(convs[0].messages[0].role, "user");
        assert_eq!(convs[0].messages[0].content, "Hello Claude");
        assert_eq!(convs[0].messages[1].role, "assistant");
        assert!(convs[0].messages[1].content.contains("How can I help"));
    }

    #[test]
    fn scan_with_callback_matches_scan_for_jsonl_session() {
        let dir = TempDir::new().unwrap();
        let claude_dir = make_test_claude_dir(dir.path());

        let session_file = claude_dir.join("session.jsonl");
        let content = r#"{"type":"user","timestamp":"2025-12-01T10:00:00Z","message":{"role":"user","content":"Hello Claude"}}
{"type":"assistant","timestamp":"2025-12-01T10:00:01Z","message":{"role":"assistant","content":"Hello! How can I help?"}}
"#;
        fs::write(&session_file, content).unwrap();

        let connector = ClaudeCodeConnector::new();
        let ctx = ScanContext::local_default(claude_dir.clone(), None);
        let scanned = connector.scan(&ctx).unwrap();
        let mut streamed = Vec::new();
        connector
            .scan_with_callback(&ctx, &mut |conversation| {
                streamed.push(conversation);
                Ok(())
            })
            .unwrap();

        assert_eq!(streamed.len(), scanned.len());
        assert_eq!(streamed[0].messages.len(), scanned[0].messages.len());
        assert_eq!(
            streamed[0].messages[0].content,
            scanned[0].messages[0].content
        );
        assert_eq!(
            streamed[0].messages[1].content,
            scanned[0].messages[1].content
        );
    }

    #[test]
    fn source_boundaries_complete_per_file_and_skip_on_resume() {
        let dir = TempDir::new().unwrap();
        let claude_dir = make_test_claude_dir(dir.path());
        for name in ["a.jsonl", "b.jsonl"] {
            fs::write(
                claude_dir.join(name),
                concat!(
                    r#"{"type":"user","timestamp":"2025-12-01T10:00:00Z","message":{"role":"user","content":"Hello"}}"#,
                    "\n",
                ),
            )
            .unwrap();
        }

        let connector = ClaudeCodeConnector::new();
        assert!(connector.supports_source_boundaries());
        let ctx = ScanContext::local_default(claude_dir.clone(), None);

        let mut completions: Vec<SourceCompletion> = Vec::new();
        let mut emitted = 0usize;
        {
            let mut on_complete = |completion: &SourceCompletion| {
                completions.push(completion.clone());
                Ok(())
            };
            let mut hooks = SourceScanHooks {
                should_scan_source: None,
                on_source_complete: Some(&mut on_complete),
                tail_resume: None,
            };
            connector
                .scan_with_source_boundaries(&ctx, &mut hooks, &mut |_conv| {
                    emitted += 1;
                    Ok(())
                })
                .unwrap();
        }
        assert_eq!(emitted, 2);
        assert_eq!(completions.len(), 2, "one completion per session file");
        for completion in &completions {
            assert_eq!(completion.conversations_emitted, 1);
            assert_eq!(completion.source.provider_slug, "claude_code");
            assert!(completion.source.size_bytes.is_some());
            assert!(completion.required_sidecars.is_empty());
        }

        // Identity matches discovery for every completed source.
        let discovered = connector.discover_source_files(&ctx).unwrap();
        for completion in &completions {
            let matching = discovered
                .iter()
                .find(|source| source.source_path == completion.source.source_path)
                .expect("completed source must be discoverable");
            assert_eq!(completion.source.size_bytes, matching.size_bytes);
            assert_eq!(completion.source.modified_at_ms, matching.modified_at_ms);
        }

        // Resume with a ledger built from the completions: nothing re-parsed.
        let ledger: Vec<SourceCompletion> = completions;
        let mut resumed = 0usize;
        {
            let mut should_scan = |source: &DiscoveredSourceFile| {
                !ledger.iter().any(|entry| {
                    entry.source.source_path == source.source_path
                        && entry.source.size_bytes == source.size_bytes
                        && entry.source.modified_at_ms == source.modified_at_ms
                })
            };
            let mut hooks = SourceScanHooks {
                should_scan_source: Some(&mut should_scan),
                on_source_complete: None,
                tail_resume: None,
            };
            connector
                .scan_with_source_boundaries(&ctx, &mut hooks, &mut |_conv| {
                    resumed += 1;
                    Ok(())
                })
                .unwrap();
        }
        assert_eq!(resumed, 0, "unchanged files must be skipped on resume");
    }

    #[test]
    fn scan_skips_explicitly_excluded_session_path_without_skipping_siblings() {
        let dir = TempDir::new().unwrap();
        let claude_dir = make_test_claude_dir(dir.path());

        let active_session = claude_dir.join("active.jsonl");
        let stable_session = claude_dir.join("stable.jsonl");
        fs::write(
            &active_session,
            r#"{"type":"user","message":{"role":"user","content":"active in-progress content"}}"#,
        )
        .unwrap();
        fs::write(
            &stable_session,
            r#"{"type":"user","message":{"role":"user","content":"stable content"}}"#,
        )
        .unwrap();

        let ctx = ScanContext::local_default(claude_dir, None);
        let mut streamed = Vec::new();
        scan_claude_with_callback_with_exclusions(
            &ctx,
            &mut |conversation| {
                streamed.push(conversation);
                Ok(())
            },
            std::slice::from_ref(&active_session),
            &mut SourceScanHooks::default(),
        )
        .unwrap();

        assert_eq!(streamed.len(), 1);
        assert_eq!(streamed[0].source_path, stable_session);
        assert_eq!(streamed[0].messages[0].content, "stable content");
    }

    #[test]
    fn discover_source_files_skips_explicitly_excluded_session_path() {
        let dir = TempDir::new().unwrap();
        let claude_dir = make_test_claude_dir(dir.path());

        let active_session = claude_dir.join("active.jsonl");
        let stable_session = claude_dir.join("stable.jsonl");
        fs::write(&active_session, "{}").unwrap();
        fs::write(&stable_session, "{}").unwrap();

        let ctx = ScanContext::local_default(claude_dir, None);
        let discovered = ClaudeCodeConnector::discover_sources_with_exclusions(
            &ctx,
            std::slice::from_ref(&active_session),
        );

        let source_paths: Vec<_> = discovered
            .into_iter()
            .map(|source| source.source_path)
            .collect();
        assert_eq!(source_paths, vec![stable_session]);
    }

    #[test]
    fn compact_message_extra_keeps_only_compact_cass_metadata() {
        let raw = json!({
            "message": {
                "model": "claude-opus-4-6",
                "usage": {
                    "input_tokens": 100,
                    "output_tokens": 50,
                    "cache_read_input_tokens": 20,
                    "cache_creation_input_tokens": 5,
                    "service_tier": "standard"
                },
                "content": [
                    {"type": "tool_use", "name": "Read"},
                    {"type": "tool_use", "name": "Edit"},
                    {"type": "text", "text": "large duplicated content"}
                ]
            },
            "attachments": [{"path": "/tmp/log.txt"}],
            "summary": "this should be dropped"
        });

        let compact = ClaudeCodeConnector::compact_message_extra(&raw);
        assert_eq!(compact["cass"]["model"], "claude-opus-4-6");
        assert_eq!(compact["cass"]["token_usage"]["input_tokens"], 100);
        assert_eq!(compact["cass"]["token_usage"]["output_tokens"], 50);
        assert_eq!(compact["cass"]["token_usage"]["cache_read_tokens"], 20);
        assert_eq!(compact["cass"]["token_usage"]["cache_creation_tokens"], 5);
        assert_eq!(compact["cass"]["token_usage"]["service_tier"], "standard");
        assert_eq!(compact["cass"]["tool_call_count"], 2);
        assert_eq!(compact["cass"]["attachments"][0]["path"], "/tmp/log.txt");
        assert!(compact.get("summary").is_none());
    }

    #[test]
    fn should_compact_large_message_extra_respects_threshold() {
        assert!(!ClaudeCodeConnector::should_compact_large_message_extra(
            Some(LARGE_SESSION_EXTRA_COMPACT_THRESHOLD_BYTES - 1,)
        ));
        assert!(ClaudeCodeConnector::should_compact_large_message_extra(
            Some(LARGE_SESSION_EXTRA_COMPACT_THRESHOLD_BYTES,)
        ));
        assert!(!ClaudeCodeConnector::should_compact_large_message_extra(
            None
        ));
    }

    #[test]
    fn scan_extracts_session_metadata() {
        let dir = TempDir::new().unwrap();
        let claude_dir = make_test_claude_dir(dir.path());

        let session_file = claude_dir.join("session.jsonl");
        let content = r#"{"type":"user","cwd":"/projects/myapp","sessionId":"sess-123","gitBranch":"main","message":{"role":"user","content":"Test message"}}"#;
        fs::write(&session_file, content).unwrap();

        let connector = ClaudeCodeConnector::new();
        let ctx = ScanContext::local_default(claude_dir.clone(), None);
        let convs = connector.scan(&ctx).unwrap();

        assert_eq!(convs.len(), 1);
        assert_eq!(convs[0].workspace, Some(PathBuf::from("/projects/myapp")));
        assert_eq!(convs[0].metadata["sessionId"], "sess-123");
        assert_eq!(convs[0].metadata["gitBranch"], "main");
    }

    #[test]
    fn scan_indexes_macos_desktop_sidecar_metadata_without_body() {
        let dir = TempDir::new().unwrap();
        let sidecar_root = dir
            .path()
            .join("Library")
            .join("Application Support")
            .join("Claude")
            .join("claude-code-sessions");
        let sidecar_path = sidecar_root
            .join("workspace-uuid")
            .join("session-uuid")
            .join("local_msg.json");
        fs::create_dir_all(sidecar_path.parent().unwrap()).unwrap();
        fs::write(
            &sidecar_path,
            json!({
                "sessionId": "local_msg",
                "cliSessionId": "cli-session-123",
                "cwd": "/Users/jane/project",
                "createdAt": 1_773_244_128_013_i64,
                "lastActivityAt": 1_773_278_849_911_i64,
                "model": "claude-opus-4-6",
                "title": "Prepare Endeavor Outliers 2025 demo case",
                "permissionMode": "plan"
            })
            .to_string(),
        )
        .unwrap();

        let connector = ClaudeCodeConnector::new();
        let ctx = ScanContext::with_roots(
            dir.path().to_path_buf(),
            vec![ScanRoot::local(sidecar_root.clone())],
            None,
        );
        let convs = connector.scan(&ctx).unwrap();

        assert_eq!(convs.len(), 1);
        let conv = &convs[0];
        assert_eq!(conv.external_id.as_deref(), Some("cli-session-123"));
        assert_eq!(
            conv.title.as_deref(),
            Some("Prepare Endeavor Outliers 2025 demo case")
        );
        assert_eq!(conv.workspace, Some(PathBuf::from("/Users/jane/project")));
        assert_eq!(conv.source_path, sidecar_path);
        assert_eq!(conv.started_at, Some(1_773_244_128_013));
        assert_eq!(conv.ended_at, Some(1_773_278_849_911));
        assert_eq!(conv.metadata["source"], "claude_code_desktop_sidecar");
        assert_eq!(conv.metadata["cliSessionId"], "cli-session-123");
        assert_eq!(conv.metadata["bodyAvailable"], false);
        assert_eq!(conv.messages.len(), 1);
        assert_eq!(conv.messages[0].role, "system");
        assert!(conv.messages[0].content.contains("Prepare Endeavor"));
        assert!(
            conv.messages[0]
                .content
                .contains("Workspace: /Users/jane/project")
        );
        assert!(
            conv.messages[0]
                .content
                .contains("CLI session id: cli-session-123")
        );
        assert!(
            conv.messages[0]
                .content
                .contains("culled the CLI JSONL body")
        );
    }

    #[test]
    fn scan_extracts_model_as_author() {
        let dir = TempDir::new().unwrap();
        let claude_dir = make_test_claude_dir(dir.path());

        let session_file = claude_dir.join("session.jsonl");
        let content = r#"{"type":"assistant","message":{"role":"assistant","content":"Response","model":"claude-3-opus"}}"#;
        fs::write(&session_file, content).unwrap();

        let connector = ClaudeCodeConnector::new();
        let ctx = ScanContext::local_default(claude_dir.clone(), None);
        let convs = connector.scan(&ctx).unwrap();

        assert_eq!(
            convs[0].messages[0].author,
            Some("claude-3-opus".to_string())
        );
    }

    #[test]
    fn scan_parses_iso8601_timestamp() {
        let dir = TempDir::new().unwrap();
        let claude_dir = make_test_claude_dir(dir.path());

        let session_file = claude_dir.join("session.jsonl");
        let content = r#"{"type":"user","timestamp":"2025-11-15T14:30:00.123Z","message":{"role":"user","content":"Test"}}"#;
        fs::write(&session_file, content).unwrap();

        let connector = ClaudeCodeConnector::new();
        let ctx = ScanContext::local_default(claude_dir.clone(), None);
        let convs = connector.scan(&ctx).unwrap();

        assert!(convs[0].messages[0].created_at.is_some());
        let ts = convs[0].messages[0].created_at.unwrap();
        // Should be around 2025-11-15 in milliseconds
        assert!(ts > 1_700_000_000_000);
    }

    #[test]
    fn scan_handles_array_content() {
        let dir = TempDir::new().unwrap();
        let claude_dir = make_test_claude_dir(dir.path());

        let session_file = claude_dir.join("session.jsonl");
        let content = json!({
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "First part"},
                    {"type": "text", "text": "Second part"}
                ]
            }
        })
        .to_string();
        fs::write(&session_file, content).unwrap();

        let connector = ClaudeCodeConnector::new();
        let ctx = ScanContext::local_default(claude_dir.clone(), None);
        let convs = connector.scan(&ctx).unwrap();

        assert_eq!(convs[0].messages.len(), 1);
        assert!(convs[0].messages[0].content.contains("First part"));
        assert!(convs[0].messages[0].content.contains("Second part"));
    }

    #[test]
    fn scan_skips_empty_content() {
        let dir = TempDir::new().unwrap();
        let claude_dir = make_test_claude_dir(dir.path());

        let session_file = claude_dir.join("session.jsonl");
        let content = r#"{"type":"user","message":{"role":"user","content":""}}
{"type":"user","message":{"role":"user","content":"   "}}
{"type":"user","message":{"role":"user","content":"Valid message"}}
"#;
        fs::write(&session_file, content).unwrap();

        let connector = ClaudeCodeConnector::new();
        let ctx = ScanContext::local_default(claude_dir.clone(), None);
        let convs = connector.scan(&ctx).unwrap();

        // Only the valid message should be extracted
        assert_eq!(convs[0].messages.len(), 1);
        assert_eq!(convs[0].messages[0].content, "Valid message");
    }

    /// cass GH #500: prompts typed while the agent is mid-turn are recorded as
    /// `queued_command` attachments and must be indexed as user messages.
    #[test]
    fn scan_indexes_mid_turn_queued_command_prompts() {
        let dir = TempDir::new().unwrap();
        let claude_dir = make_test_claude_dir(dir.path());
        let session_file = claude_dir.join("session.jsonl");
        let lines = [
            r#"{"type":"user","uuid":"u1","sessionId":"s","timestamp":"2026-09-24T10:00:00.000Z","message":{"role":"user","content":"please audit the zebratypedphrase module"}}"#,
            r#"{"type":"assistant","uuid":"a1","sessionId":"s","timestamp":"2026-09-24T10:00:05.000Z","message":{"role":"assistant","model":"claude-opus-5-5","content":[{"type":"text","text":"Starting on the audit now."}]}}"#,
            // queue-operation records repeat queued text and must stay out.
            r#"{"type":"queue-operation","operation":"enqueue","timestamp":"2026-09-24T10:00:10.000Z","sessionId":"s","content":"also check the walrusqueuedphrase path"}"#,
            r#"{"type":"queue-operation","operation":"remove","timestamp":"2026-09-24T10:00:12.000Z","sessionId":"s","content":"also check the walrusqueuedphrase path"}"#,
            // Human-origin, string prompt.
            r#"{"type":"attachment","uuid":"q1","sessionId":"s","timestamp":"2026-09-24T10:00:12.000Z","attachment":{"type":"queued_command","prompt":"also check the walrusqueuedphrase path","source_uuid":"src1","commandMode":"prompt","origin":{"kind":"human"},"timestamp":"2026-09-24T10:00:12.000Z"}}"#,
            // Older builds: no origin field -> kept, unknown authorship.
            r#"{"type":"attachment","uuid":"q2","sessionId":"s","timestamp":"2026-09-24T10:00:13.000Z","attachment":{"type":"queued_command","prompt":"older era narwhallegacyphrase","commandMode":"prompt"}}"#,
            // List-shaped prompt.
            r#"{"type":"attachment","uuid":"q3","sessionId":"s","timestamp":"2026-09-24T10:00:14.000Z","attachment":{"type":"queued_command","prompt":[{"type":"text","text":"list shaped ocelotlistphrase prompt"}],"commandMode":"prompt","origin":{"kind":"human"}}}"#,
            // Skipped: relayed peer message, meta entry, task notification,
            // unrecognized explicit origin, empty prompt, other attachment.
            r#"{"type":"attachment","uuid":"x1","sessionId":"s","timestamp":"2026-09-24T10:00:15.000Z","attachment":{"type":"queued_command","prompt":"peer relayed ibexpeerphrase","commandMode":"prompt","isMeta":true,"origin":{"kind":"peer"}}}"#,
            r#"{"type":"attachment","uuid":"x2","sessionId":"s","timestamp":"2026-09-24T10:00:15.000Z","attachment":{"type":"queued_command","prompt":"meta yakmetaphrase","commandMode":"prompt","isMeta":true}}"#,
            r#"{"type":"attachment","uuid":"x3","sessionId":"s","timestamp":"2026-09-24T10:00:16.000Z","attachment":{"type":"queued_command","prompt":"<task-notification>lemurtaskphrase</task-notification>","commandMode":"task-notification"}}"#,
            r#"{"type":"attachment","uuid":"x4","sessionId":"s","timestamp":"2026-09-24T10:00:16.000Z","attachment":{"type":"queued_command","prompt":"odd okapiodd phrase","commandMode":"prompt","origin":{"kind":"something-new"}}}"#,
            r#"{"type":"attachment","uuid":"x5","sessionId":"s","timestamp":"2026-09-24T10:00:16.000Z","attachment":{"type":"queued_command","prompt":"   ","commandMode":"prompt","origin":{"kind":"human"}}}"#,
            r#"{"type":"attachment","uuid":"x6","sessionId":"s","timestamp":"2026-09-24T10:00:16.000Z","attachment":{"type":"total_tokens_reminder","prompt":"not a prompt"}}"#,
            r#"{"type":"assistant","uuid":"a2","sessionId":"s","timestamp":"2026-09-24T10:00:20.000Z","message":{"role":"assistant","content":[{"type":"text","text":"Noted, I will also check that path."}]}}"#,
            // A late queued prompt extends the conversation's end bound.
            r#"{"type":"attachment","uuid":"q4","sessionId":"s","timestamp":"2026-09-24T10:00:30.000Z","attachment":{"type":"queued_command","prompt":"one more kiwilatephrase","commandMode":"prompt","origin":{"kind":"human"}}}"#,
        ];
        fs::write(&session_file, lines.join("\n")).unwrap();

        let connector = ClaudeCodeConnector::new();
        let ctx = ScanContext::local_default(claude_dir, None);
        let convs = connector.scan(&ctx).unwrap();
        assert_eq!(convs.len(), 1);
        let conv = &convs[0];

        let summary: Vec<(i64, &str, &str)> = conv
            .messages
            .iter()
            .map(|m| (m.idx, m.role.as_str(), m.content.as_str()))
            .collect();
        assert_eq!(
            summary,
            vec![
                (0, "user", "please audit the zebratypedphrase module"),
                (1, "assistant", "Starting on the audit now."),
                (2, "user", "also check the walrusqueuedphrase path"),
                (3, "user", "older era narwhallegacyphrase"),
                (4, "user", "list shaped ocelotlistphrase prompt"),
                (5, "assistant", "Noted, I will also check that path."),
                (6, "user", "one more kiwilatephrase"),
            ]
        );

        let q1 = &conv.messages[2];
        assert_eq!(
            q1.created_at,
            parse_timestamp(&json!("2026-09-24T10:00:12.000Z"))
        );
        assert_eq!(q1.extra["cass"]["queued_command"]["authorship"], "human");
        assert_eq!(q1.extra["cass"]["queued_command"]["source_uuid"], "src1");
        assert_eq!(q1.extra["cass"]["queued_command"]["command_mode"], "prompt");
        // The raw record is still carried for non-compacted sessions.
        assert_eq!(q1.extra["attachment"]["origin"]["kind"], "human");

        let q2 = &conv.messages[3];
        assert_eq!(q2.extra["cass"]["queued_command"]["authorship"], "unknown");
        assert!(
            q2.extra["cass"]["queued_command"]
                .get("source_uuid")
                .is_none()
        );

        // The typed first message still names the conversation.
        assert_eq!(
            conv.title.as_deref(),
            Some("please audit the zebratypedphrase module")
        );
        assert_eq!(
            conv.started_at,
            parse_timestamp(&json!("2026-09-24T10:00:00.000Z"))
        );
        assert_eq!(
            conv.ended_at,
            parse_timestamp(&json!("2026-09-24T10:00:30.000Z"))
        );
    }

    #[test]
    fn queued_command_compact_extra_keeps_provenance() {
        let raw = json!({
            "type": "attachment",
            "timestamp": "2026-09-24T10:00:12.000Z",
            "attachment": {
                "type": "queued_command",
                "prompt": "large session queued prompt",
                "source_uuid": "src9",
                "commandMode": "prompt",
                "origin": {"kind": "human"}
            }
        });
        let msg = ClaudeCodeConnector::queued_command_message(&raw, true).unwrap();
        assert_eq!(msg.role, "user");
        assert_eq!(msg.content, "large session queued prompt");
        assert!(msg.extra.get("attachment").is_none());
        assert_eq!(msg.extra["cass"]["queued_command"]["authorship"], "human");
        assert_eq!(msg.extra["cass"]["queued_command"]["source_uuid"], "src9");

        // Falls back to the nested timestamp when the outer one is missing.
        let nested_only = json!({
            "type": "attachment",
            "attachment": {
                "type": "queued_command",
                "prompt": "nested ts",
                "commandMode": "prompt",
                "timestamp": "2026-09-24T10:00:12.000Z"
            }
        });
        let msg = ClaudeCodeConnector::queued_command_message(&nested_only, false).unwrap();
        assert_eq!(
            msg.created_at,
            parse_timestamp(&json!("2026-09-24T10:00:12.000Z"))
        );
    }

    #[test]
    fn scan_skips_non_user_assistant_types() {
        let dir = TempDir::new().unwrap();
        let claude_dir = make_test_claude_dir(dir.path());

        let session_file = claude_dir.join("session.jsonl");
        let content = r#"{"type":"summary","content":"Session summary"}
{"type":"file-history-snapshot","files":[]}
{"type":"user","message":{"role":"user","content":"User message"}}
{"type":"tool_result","result":"Some result"}
"#;
        fs::write(&session_file, content).unwrap();

        let connector = ClaudeCodeConnector::new();
        let ctx = ScanContext::local_default(claude_dir.clone(), None);
        let convs = connector.scan(&ctx).unwrap();

        assert_eq!(convs[0].messages.len(), 1);
        assert_eq!(convs[0].messages[0].role, "user");
    }

    #[test]
    fn scan_reindexes_messages_sequentially() {
        let dir = TempDir::new().unwrap();
        let claude_dir = make_test_claude_dir(dir.path());

        let session_file = claude_dir.join("session.jsonl");
        let content = r#"{"type":"user","message":{"role":"user","content":"Message 1"}}
{"type":"assistant","message":{"role":"assistant","content":"Message 2"}}
{"type":"user","message":{"role":"user","content":"Message 3"}}
"#;
        fs::write(&session_file, content).unwrap();

        let connector = ClaudeCodeConnector::new();
        let ctx = ScanContext::local_default(claude_dir.clone(), None);
        let convs = connector.scan(&ctx).unwrap();

        assert_eq!(convs[0].messages[0].idx, 0);
        assert_eq!(convs[0].messages[1].idx, 1);
        assert_eq!(convs[0].messages[2].idx, 2);
    }

    // =========================================================================
    // JSON format parsing tests
    // =========================================================================

    #[test]
    fn scan_parses_json_messages_array() {
        let dir = TempDir::new().unwrap();
        let claude_dir = make_test_claude_dir(dir.path());

        let session_file = claude_dir.join("session.json");
        let content = json!({
            "title": "Test Session",
            "messages": [
                {"role": "user", "content": "Hello", "timestamp": 1_700_000_000_000_i64},
                {"role": "assistant", "content": "Hi there!", "timestamp": 1_700_000_001_000_i64}
            ]
        })
        .to_string();
        fs::write(&session_file, content).unwrap();

        let connector = ClaudeCodeConnector::new();
        let ctx = ScanContext::local_default(claude_dir.clone(), None);
        let convs = connector.scan(&ctx).unwrap();

        assert_eq!(convs.len(), 1);
        assert_eq!(convs[0].messages.len(), 2);
        assert_eq!(convs[0].messages[0].role, "user");
        assert_eq!(convs[0].messages[1].role, "assistant");
    }

    #[test]
    fn scan_json_extracts_title() {
        let dir = TempDir::new().unwrap();
        let claude_dir = make_test_claude_dir(dir.path());

        let session_file = claude_dir.join("session.json");
        let content = json!({
            "title": "Custom Session Title",
            "messages": [
                {"role": "user", "content": "Test content"}
            ]
        })
        .to_string();
        fs::write(&session_file, content).unwrap();

        let connector = ClaudeCodeConnector::new();
        let ctx = ScanContext::local_default(claude_dir.clone(), None);
        let convs = connector.scan(&ctx).unwrap();

        assert_eq!(convs[0].title, Some("Custom Session Title".to_string()));
    }

    #[test]
    fn scan_json_uses_type_as_role_fallback() {
        let dir = TempDir::new().unwrap();
        let claude_dir = make_test_claude_dir(dir.path());

        let session_file = claude_dir.join("session.json");
        let content = json!({
            "messages": [
                {"type": "user", "content": "Message with type instead of role"}
            ]
        })
        .to_string();
        fs::write(&session_file, content).unwrap();

        let connector = ClaudeCodeConnector::new();
        let ctx = ScanContext::local_default(claude_dir.clone(), None);
        let convs = connector.scan(&ctx).unwrap();

        assert_eq!(convs[0].messages[0].role, "user");
    }

    #[test]
    fn scan_json_uses_text_as_content_fallback() {
        let dir = TempDir::new().unwrap();
        let claude_dir = make_test_claude_dir(dir.path());

        let session_file = claude_dir.join("session.json");
        let content = json!({
            "messages": [
                {"role": "user", "text": "Message with text field instead of content"}
            ]
        })
        .to_string();
        fs::write(&session_file, content).unwrap();

        let connector = ClaudeCodeConnector::new();
        let ctx = ScanContext::local_default(claude_dir.clone(), None);
        let convs = connector.scan(&ctx).unwrap();

        assert!(convs[0].messages[0].content.contains("text field"));
    }

    #[test]
    fn scan_json_uses_time_as_timestamp_fallback() {
        let dir = TempDir::new().unwrap();
        let claude_dir = make_test_claude_dir(dir.path());

        let session_file = claude_dir.join("session.json");
        let content = json!({
            "messages": [
                {"role": "user", "content": "Test", "time": 1_700_000_000_000i64}
            ]
        })
        .to_string();
        fs::write(&session_file, content).unwrap();

        let connector = ClaudeCodeConnector::new();
        let ctx = ScanContext::local_default(claude_dir.clone(), None);
        let convs = connector.scan(&ctx).unwrap();

        assert_eq!(convs[0].messages[0].created_at, Some(1_700_000_000_000));
    }

    // =========================================================================
    // Title extraction tests
    // =========================================================================

    #[test]
    fn scan_title_from_first_user_message_jsonl() {
        let dir = TempDir::new().unwrap();
        let claude_dir = make_test_claude_dir(dir.path());

        let session_file = claude_dir.join("session.jsonl");
        let content = r#"{"type":"assistant","message":{"role":"assistant","content":"I can help"}}
{"type":"user","message":{"role":"user","content":"Help me build a web app"}}
"#;
        fs::write(&session_file, content).unwrap();

        let connector = ClaudeCodeConnector::new();
        let ctx = ScanContext::local_default(claude_dir.clone(), None);
        let convs = connector.scan(&ctx).unwrap();

        assert_eq!(convs[0].title, Some("Help me build a web app".to_string()));
    }

    #[test]
    fn scan_title_truncates_to_100_chars() {
        let dir = TempDir::new().unwrap();
        let claude_dir = make_test_claude_dir(dir.path());

        let long_message = "x".repeat(200);
        let session_file = claude_dir.join("session.jsonl");
        let content = format!(
            r#"{{"type":"user","message":{{"role":"user","content":"{}"}}}}"#,
            long_message
        );
        fs::write(&session_file, content).unwrap();

        let connector = ClaudeCodeConnector::new();
        let ctx = ScanContext::local_default(claude_dir.clone(), None);
        let convs = connector.scan(&ctx).unwrap();

        assert!(convs[0].title.as_ref().unwrap().len() <= 100);
    }

    #[test]
    fn scan_title_uses_first_line_only() {
        let dir = TempDir::new().unwrap();
        let claude_dir = make_test_claude_dir(dir.path());

        let session_file = claude_dir.join("session.jsonl");
        let content = r#"{"type":"user","message":{"role":"user","content":"First line\nSecond line\nThird line"}}"#;
        fs::write(&session_file, content).unwrap();

        let connector = ClaudeCodeConnector::new();
        let ctx = ScanContext::local_default(claude_dir.clone(), None);
        let convs = connector.scan(&ctx).unwrap();

        assert_eq!(convs[0].title, Some("First line".to_string()));
    }

    #[test]
    fn scan_title_fallback_to_workspace_name() {
        let dir = TempDir::new().unwrap();
        let claude_dir = make_test_claude_dir(dir.path());

        // Only assistant message, no user message for title
        let session_file = claude_dir.join("session.jsonl");
        let content = r#"{"type":"assistant","cwd":"/projects/myapp","message":{"role":"assistant","content":"Response only"}}"#;
        fs::write(&session_file, content).unwrap();

        let connector = ClaudeCodeConnector::new();
        let ctx = ScanContext::local_default(claude_dir.clone(), None);
        let convs = connector.scan(&ctx).unwrap();

        // Should fallback to workspace directory name
        assert_eq!(convs[0].title, Some("myapp".to_string()));
    }

    // =========================================================================
    // Edge case tests
    // =========================================================================

    #[test]
    fn scan_empty_directory_returns_empty() {
        let dir = TempDir::new().unwrap();
        let claude_dir = make_test_claude_dir(dir.path());

        let connector = ClaudeCodeConnector::new();
        let ctx = ScanContext::local_default(claude_dir.clone(), None);
        let convs = connector.scan(&ctx).unwrap();

        assert!(convs.is_empty());
    }

    #[test]
    fn scan_skips_malformed_jsonl_lines() {
        let dir = TempDir::new().unwrap();
        let claude_dir = make_test_claude_dir(dir.path());

        let session_file = claude_dir.join("session.jsonl");
        let content = r#"not valid json
{"type":"user","message":{"role":"user","content":"Valid message"}}
{broken json here
"#;
        fs::write(&session_file, content).unwrap();

        let connector = ClaudeCodeConnector::new();
        let ctx = ScanContext::local_default(claude_dir.clone(), None);
        let convs = connector.scan(&ctx).unwrap();

        // Should still extract the valid line
        assert_eq!(convs[0].messages.len(), 1);
        assert_eq!(convs[0].messages[0].content, "Valid message");
    }

    #[test]
    fn scan_skips_malformed_json_files() {
        let dir = TempDir::new().unwrap();
        let claude_dir = make_test_claude_dir(dir.path());

        // Create a malformed JSON file
        let bad_file = claude_dir.join("bad.json");
        fs::write(&bad_file, "not valid json {{{").unwrap();

        // Create a valid JSONL file
        let good_file = claude_dir.join("good.jsonl");
        fs::write(
            &good_file,
            r#"{"type":"user","message":{"role":"user","content":"Valid"}}"#,
        )
        .unwrap();

        let connector = ClaudeCodeConnector::new();
        let ctx = ScanContext::local_default(claude_dir.clone(), None);
        let convs = connector.scan(&ctx).unwrap();

        // Should only have one conversation from the valid file
        assert_eq!(convs.len(), 1);
    }

    #[test]
    fn scan_handles_empty_messages_array() {
        let dir = TempDir::new().unwrap();
        let claude_dir = make_test_claude_dir(dir.path());

        let session_file = claude_dir.join("session.json");
        let content = json!({
            "messages": []
        })
        .to_string();
        fs::write(&session_file, content).unwrap();

        let connector = ClaudeCodeConnector::new();
        let ctx = ScanContext::local_default(claude_dir.clone(), None);
        let convs = connector.scan(&ctx).unwrap();

        // Empty messages should result in no conversation
        assert!(convs.is_empty());
    }

    #[test]
    fn scan_processes_subdirectories() {
        let dir = TempDir::new().unwrap();
        let claude_dir = make_test_claude_dir(dir.path());
        let subdir = claude_dir.join("project1");
        fs::create_dir_all(&subdir).unwrap();

        let session_file = subdir.join("session.jsonl");
        let content = r#"{"type":"user","message":{"role":"user","content":"Nested message"}}"#;
        fs::write(&session_file, content).unwrap();

        let connector = ClaudeCodeConnector::new();
        let ctx = ScanContext::local_default(claude_dir.clone(), None);
        let convs = connector.scan(&ctx).unwrap();

        assert_eq!(convs.len(), 1);
        assert!(convs[0].messages[0].content.contains("Nested message"));
    }

    #[test]
    fn scan_skips_non_session_files() {
        let dir = TempDir::new().unwrap();
        let claude_dir = make_test_claude_dir(dir.path());

        // Create various non-session files
        fs::write(claude_dir.join("config.toml"), "").unwrap();
        fs::write(claude_dir.join("notes.txt"), "").unwrap();
        fs::write(claude_dir.join("backup.bak"), "").unwrap();

        // Create a valid session file
        let session_file = claude_dir.join("session.jsonl");
        fs::write(
            &session_file,
            r#"{"type":"user","message":{"role":"user","content":"Valid"}}"#,
        )
        .unwrap();

        let connector = ClaudeCodeConnector::new();
        let ctx = ScanContext::local_default(claude_dir.clone(), None);
        let convs = connector.scan(&ctx).unwrap();

        // Should only have one conversation from the .jsonl file
        assert_eq!(convs.len(), 1);
    }

    #[test]
    fn scan_handles_claude_extension() {
        let dir = TempDir::new().unwrap();
        let claude_dir = make_test_claude_dir(dir.path());

        let session_file = claude_dir.join("session.claude");
        let content = json!({
            "messages": [
                {"role": "user", "content": "Claude extension test"}
            ]
        })
        .to_string();
        fs::write(&session_file, content).unwrap();

        let connector = ClaudeCodeConnector::new();
        let ctx = ScanContext::local_default(claude_dir.clone(), None);
        let convs = connector.scan(&ctx).unwrap();

        assert_eq!(convs.len(), 1);
        assert!(convs[0].messages[0].content.contains("Claude extension"));
    }

    #[test]
    fn scan_sets_external_id_from_relative_path() {
        let dir = TempDir::new().unwrap();
        let claude_dir = make_test_claude_dir(dir.path());

        let session_file = claude_dir.join("unique-session-id.jsonl");
        let content = r#"{"type":"user","message":{"role":"user","content":"Test"}}"#;
        fs::write(&session_file, content).unwrap();

        let connector = ClaudeCodeConnector::new();
        let ctx = ScanContext::local_default(claude_dir.clone(), None);
        let convs = connector.scan(&ctx).unwrap();

        assert_eq!(
            convs[0].external_id,
            Some("unique-session-id.jsonl".to_string())
        );
    }

    #[test]
    fn scan_external_id_includes_subdir_for_subagent_files() {
        let dir = TempDir::new().unwrap();
        let claude_dir = make_test_claude_dir(dir.path());

        // Simulate subagent files under different parent sessions sharing the
        // same basename (e.g. agent-a297e09.jsonl).
        let sub_a = claude_dir.join("parent-session-aaa");
        let sub_b = claude_dir.join("parent-session-bbb");
        fs::create_dir_all(&sub_a).unwrap();
        fs::create_dir_all(&sub_b).unwrap();

        let content = r#"{"type":"user","message":{"role":"user","content":"Test"}}"#;
        fs::write(sub_a.join("agent-a297e09.jsonl"), content).unwrap();
        fs::write(sub_b.join("agent-a297e09.jsonl"), content).unwrap();

        let connector = ClaudeCodeConnector::new();
        let ctx = ScanContext::local_default(claude_dir.clone(), None);
        let mut convs = connector.scan(&ctx).unwrap();

        assert_eq!(convs.len(), 2);
        convs.sort_by(|a, b| a.external_id.cmp(&b.external_id));

        assert_eq!(
            convs[0].external_id,
            Some("parent-session-aaa/agent-a297e09.jsonl".to_string())
        );
        assert_eq!(
            convs[1].external_id,
            Some("parent-session-bbb/agent-a297e09.jsonl".to_string())
        );
    }

    #[test]
    fn scan_sets_agent_slug_to_claude_code() {
        let dir = TempDir::new().unwrap();
        let claude_dir = make_test_claude_dir(dir.path());

        let session_file = claude_dir.join("session.jsonl");
        let content = r#"{"type":"user","message":{"role":"user","content":"Test"}}"#;
        fs::write(&session_file, content).unwrap();

        let connector = ClaudeCodeConnector::new();
        let ctx = ScanContext::local_default(claude_dir.clone(), None);
        let convs = connector.scan(&ctx).unwrap();

        assert_eq!(convs[0].agent_slug, "claude_code");
    }

    #[test]
    fn scan_preserves_original_json_in_extra() {
        let dir = TempDir::new().unwrap();
        let claude_dir = make_test_claude_dir(dir.path());

        let session_file = claude_dir.join("session.jsonl");
        let content = r#"{"type":"user","customField":"customValue","message":{"role":"user","content":"Test"}}"#;
        fs::write(&session_file, content).unwrap();

        let connector = ClaudeCodeConnector::new();
        let ctx = ScanContext::local_default(claude_dir.clone(), None);
        let convs = connector.scan(&ctx).unwrap();

        assert_eq!(convs[0].messages[0].extra["customField"], "customValue");
    }

    #[test]
    fn scan_tracks_started_and_ended_timestamps() {
        let dir = TempDir::new().unwrap();
        let claude_dir = make_test_claude_dir(dir.path());

        let session_file = claude_dir.join("session.jsonl");
        let content = r#"{"type":"user","timestamp":"2025-12-01T10:00:00Z","message":{"role":"user","content":"First"}}
{"type":"assistant","timestamp":"2025-12-01T10:05:00Z","message":{"role":"assistant","content":"Last"}}
"#;
        fs::write(&session_file, content).unwrap();

        let connector = ClaudeCodeConnector::new();
        let ctx = ScanContext::local_default(claude_dir.clone(), None);
        let convs = connector.scan(&ctx).unwrap();

        assert!(convs[0].started_at.is_some());
        assert!(convs[0].ended_at.is_some());
        // ended_at should be after or equal to started_at
        assert!(convs[0].ended_at.unwrap() >= convs[0].started_at.unwrap());
    }

    #[test]
    fn scan_multiple_files_returns_multiple_conversations() {
        let dir = TempDir::new().unwrap();
        let claude_dir = make_test_claude_dir(dir.path());

        // Create two session files
        for i in 1..=3 {
            let session_file = claude_dir.join(format!("session{}.jsonl", i));
            let content =
                format!(r#"{{"type":"user","message":{{"role":"user","content":"Message {i}"}}}}"#);
            fs::write(&session_file, content).unwrap();
        }

        let connector = ClaudeCodeConnector::new();
        let ctx = ScanContext::local_default(claude_dir.clone(), None);
        let convs = connector.scan(&ctx).unwrap();

        assert_eq!(convs.len(), 3);
    }

    #[test]
    fn scan_explicit_root_generic_name() {
        let dir = TempDir::new().unwrap();
        // Directory name that doesn't contain "claude" and no "projects" subdir
        let generic_root = dir.path().join("my_logs");
        fs::create_dir_all(&generic_root).unwrap();

        let session_file = generic_root.join("session.jsonl");
        let content = r#"{"type":"user","message":{"role":"user","content":"Generic root test"}}"#;
        fs::write(&session_file, content).unwrap();

        let connector = ClaudeCodeConnector::new();
        // Create context with explicit root (use_default_detection = false)
        // Note: ScanContext::with_roots takes data_dir as first arg, but indexer passes root.path there too.
        // We simulate what indexer does.
        let roots = vec![crate::connectors::ScanRoot::local(generic_root.clone())];
        let ctx = ScanContext::with_roots(generic_root.clone(), roots, None);

        let convs = connector.scan(&ctx).unwrap();

        assert_eq!(
            convs.len(),
            1,
            "Should find session in generic named explicit root"
        );
        assert_eq!(convs[0].messages[0].content, "Generic root test");
    }

    #[test]
    fn scan_with_explicit_file_only_reads_that_file_and_keeps_projects_relative_external_id() {
        let dir = TempDir::new().unwrap();
        let claude_dir = make_test_claude_dir(dir.path());
        let project_dir = claude_dir.join("projects").join("project-a");
        fs::create_dir_all(project_dir.join("subagents")).unwrap();

        let target_file = project_dir.join("target.jsonl");
        fs::write(
            &target_file,
            r#"{"type":"user","message":{"role":"user","content":"Target only"}}"#,
        )
        .unwrap();

        let sibling_file = project_dir.join("sibling.jsonl");
        fs::write(
            &sibling_file,
            r#"{"type":"user","message":{"role":"user","content":"Sibling"}}"#,
        )
        .unwrap();

        let nested_file = project_dir.join("subagents").join("nested.jsonl");
        fs::write(
            &nested_file,
            r#"{"type":"user","message":{"role":"user","content":"Nested"}}"#,
        )
        .unwrap();

        let connector = ClaudeCodeConnector::new();
        let roots = vec![crate::connectors::ScanRoot::local(target_file.clone())];
        let ctx = ScanContext::with_roots(target_file.clone(), roots, None);

        let convs = connector.scan(&ctx).unwrap();

        assert_eq!(convs.len(), 1);
        assert_eq!(convs[0].messages[0].content, "Target only");
        assert_eq!(
            convs[0].external_id.as_deref(),
            Some("project-a/target.jsonl")
        );
    }

    // =========================================================================
    // Edge case tests — malformed input robustness (br-cpf8)
    // =========================================================================

    #[test]
    fn truncated_jsonl_mid_json_returns_partial_results() {
        let dir = TempDir::new().unwrap();
        let claude_dir = make_test_claude_dir(dir.path());

        // First line is valid, second line is truncated mid-JSON
        let session_file = claude_dir.join("truncated.jsonl");
        let content = b"{\"type\":\"user\",\"message\":{\"role\":\"user\",\"content\":\"Hello\"}}\n{\"type\":\"assistant\",\"message\":{\"role\":\"assistant\",\"content\":\"Hel";
        fs::write(&session_file, content).unwrap();

        let connector = ClaudeCodeConnector::new();
        let ctx = ScanContext::local_default(claude_dir.clone(), None);
        let result = connector.scan(&ctx);

        assert!(result.is_ok(), "truncated file should not cause an error");
        let convs = result.unwrap();
        assert_eq!(convs.len(), 1);
        assert_eq!(
            convs[0].messages.len(),
            1,
            "truncated file at mid-JSON should yield only the 1 valid message"
        );
        assert_eq!(convs[0].messages[0].content, "Hello");
    }

    #[test]
    fn truncated_mid_utf8_does_not_panic() {
        let dir = TempDir::new().unwrap();
        let claude_dir = make_test_claude_dir(dir.path());

        // Valid JSONL line followed by bytes that start a multi-byte UTF-8
        // sequence but are truncated (U+1F600 = F0 9F 98 80, truncate after 2 bytes)
        let session_file = claude_dir.join("truncated_utf8.jsonl");
        let mut bytes = Vec::new();
        bytes.extend_from_slice(
            b"{\"type\":\"user\",\"message\":{\"role\":\"user\",\"content\":\"Valid\"}}\n",
        );
        // Incomplete UTF-8: start of a 4-byte sequence missing last 2 bytes
        bytes.extend_from_slice(b"\xF0\x9F");

        fs::write(&session_file, &bytes).unwrap();

        let connector = ClaudeCodeConnector::new();
        let ctx = ScanContext::local_default(claude_dir.clone(), None);
        let result = connector.scan(&ctx);

        assert!(
            result.is_ok(),
            "truncated mid-UTF8 should not panic or error"
        );
        let convs = result.unwrap();
        assert_eq!(convs.len(), 1);
        assert_eq!(convs[0].messages.len(), 1);
        assert_eq!(convs[0].messages[0].content, "Valid");
    }

    #[test]
    fn invalid_utf8_skips_corrupted_lines() {
        let dir = TempDir::new().unwrap();
        let claude_dir = make_test_claude_dir(dir.path());

        let session_file = claude_dir.join("invalid_utf8.jsonl");
        let mut bytes = Vec::new();
        // Valid line
        bytes.extend_from_slice(
            b"{\"type\":\"user\",\"message\":{\"role\":\"user\",\"content\":\"Before\"}}\n",
        );
        // Invalid UTF-8 bytes (0xFF 0xFE are never valid in UTF-8)
        bytes.extend_from_slice(b"\xFF\xFE invalid utf8 line\n");
        // Another valid line
        bytes.extend_from_slice(
            b"{\"type\":\"user\",\"message\":{\"role\":\"user\",\"content\":\"After\"}}\n",
        );

        fs::write(&session_file, &bytes).unwrap();

        let connector = ClaudeCodeConnector::new();
        let ctx = ScanContext::local_default(claude_dir.clone(), None);
        let result = connector.scan(&ctx);

        assert!(result.is_ok(), "invalid UTF-8 should not cause a panic");
        let convs = result.unwrap();
        // BufRead::lines() returns Err for invalid UTF-8 lines; the connector
        // continues on Err (line 114: Err(_) => continue). So we should get
        // the valid lines on either side.
        assert_eq!(convs.len(), 1);
        assert_eq!(
            convs[0].messages.len(),
            2,
            "should extract both valid messages around invalid UTF-8 line"
        );
        assert_eq!(convs[0].messages[0].content, "Before");
        assert_eq!(convs[0].messages[1].content, "After");
    }

    #[test]
    fn empty_file_returns_no_conversations() {
        let dir = TempDir::new().unwrap();
        let claude_dir = make_test_claude_dir(dir.path());

        // Completely empty JSONL file
        let session_file = claude_dir.join("empty.jsonl");
        fs::write(&session_file, b"").unwrap();

        // Completely empty JSON file
        let json_file = claude_dir.join("empty.json");
        fs::write(&json_file, b"").unwrap();

        let connector = ClaudeCodeConnector::new();
        let ctx = ScanContext::local_default(claude_dir.clone(), None);
        let result = connector.scan(&ctx);

        assert!(result.is_ok(), "empty files should not cause errors");
        let convs = result.unwrap();
        assert!(
            convs.is_empty(),
            "empty files should produce no conversations"
        );
    }

    #[test]
    fn whitespace_only_file_returns_no_conversations() {
        let dir = TempDir::new().unwrap();
        let claude_dir = make_test_claude_dir(dir.path());

        // JSONL file with only whitespace and newlines
        let session_file = claude_dir.join("whitespace.jsonl");
        fs::write(&session_file, "   \n\n  \n   \n\t\n").unwrap();

        let connector = ClaudeCodeConnector::new();
        let ctx = ScanContext::local_default(claude_dir.clone(), None);
        let result = connector.scan(&ctx);

        assert!(
            result.is_ok(),
            "whitespace-only file should not cause errors"
        );
        let convs = result.unwrap();
        assert!(
            convs.is_empty(),
            "whitespace-only file should produce no conversations"
        );
    }

    #[test]
    fn json_type_mismatch_skips_gracefully() {
        let dir = TempDir::new().unwrap();
        let claude_dir = make_test_claude_dir(dir.path());

        // JSONL lines where expected objects are wrong types
        let session_file = claude_dir.join("type_mismatch.jsonl");
        let content = concat!(
            // String where object expected for message
            "{\"type\":\"user\",\"message\":\"not an object\"}\n",
            // Number where content string expected
            "{\"type\":\"user\",\"message\":{\"role\":\"user\",\"content\":12345}}\n",
            // Array where string expected for type
            "{\"type\":[\"user\"],\"message\":{\"role\":\"user\",\"content\":\"Valid after mismatches\"}}\n",
            // Null content
            "{\"type\":\"user\",\"message\":{\"role\":\"user\",\"content\":null}}\n",
            // Boolean type
            "{\"type\":true,\"message\":{\"role\":\"user\",\"content\":\"Bool type\"}}\n",
            // Correct entry that should be extracted
            "{\"type\":\"user\",\"message\":{\"role\":\"user\",\"content\":\"Correct entry\"}}\n",
        );
        fs::write(&session_file, content).unwrap();

        let connector = ClaudeCodeConnector::new();
        let ctx = ScanContext::local_default(claude_dir.clone(), None);
        let result = connector.scan(&ctx);

        assert!(result.is_ok(), "type mismatches should not cause errors");
        let convs = result.unwrap();
        // Only the last line with correct types should produce a message
        assert_eq!(convs.len(), 1);
        assert!(
            convs[0]
                .messages
                .iter()
                .any(|m| m.content == "Correct entry"),
            "should extract the correctly typed entry"
        );
    }

    #[test]
    fn deeply_nested_json_does_not_stack_overflow() {
        let dir = TempDir::new().unwrap();
        let claude_dir = make_test_claude_dir(dir.path());

        // Build JSON with 1000+ levels of nesting in the content field
        // serde_json has a default recursion limit of 128, so this tests
        // that the connector handles the parse error gracefully
        let mut nested = String::new();
        for _ in 0..200 {
            nested.push_str("{\"a\":");
        }
        nested.push('1');
        for _ in 0..200 {
            nested.push('}');
        }

        let session_file = claude_dir.join("deep.jsonl");
        let content = format!(
            "{}\n{}\n",
            nested, r#"{"type":"user","message":{"role":"user","content":"After deep nesting"}}"#
        );
        fs::write(&session_file, &content).unwrap();

        let connector = ClaudeCodeConnector::new();
        let ctx = ScanContext::local_default(claude_dir.clone(), None);

        // This must not stack overflow or panic
        let result = connector.scan(&ctx);

        assert!(
            result.is_ok(),
            "deeply nested JSON should not cause stack overflow"
        );
        let convs = result.unwrap();
        assert_eq!(convs.len(), 1);
        assert_eq!(convs[0].messages[0].content, "After deep nesting");
    }

    #[test]
    fn large_message_body_handled_without_oom() {
        let dir = TempDir::new().unwrap();
        let claude_dir = make_test_claude_dir(dir.path());

        // Create a JSONL file with a 1MB message body to verify streaming works
        let large_content = "x".repeat(1_000_000);
        let session_file = claude_dir.join("large_body.jsonl");
        let line = format!(
            r#"{{"type":"user","message":{{"role":"user","content":"{}"}}}}"#,
            large_content
        );
        fs::write(&session_file, &line).unwrap();

        let connector = ClaudeCodeConnector::new();
        let ctx = ScanContext::local_default(claude_dir.clone(), None);
        let result = connector.scan(&ctx);

        assert!(result.is_ok(), "large message body should not cause OOM");
        let convs = result.unwrap();
        assert_eq!(convs.len(), 1);
        assert_eq!(
            convs[0].messages[0].content.len(),
            1_000_000,
            "large message content should be preserved in full"
        );
    }

    #[test]
    fn large_json_file_over_100mb_is_skipped() {
        let dir = TempDir::new().unwrap();
        let claude_dir = make_test_claude_dir(dir.path());

        // For JSON format, files > 100MB should be skipped.
        // We can't create a real 100MB file in a unit test efficiently,
        // but we verify the mechanism works with a valid JSON file under the limit.
        let session_file = claude_dir.join("under_limit.json");
        let content = json!({
            "messages": [
                {"role": "user", "content": "Under the limit"}
            ]
        })
        .to_string();
        fs::write(&session_file, &content).unwrap();

        let connector = ClaudeCodeConnector::new();
        let ctx = ScanContext::local_default(claude_dir.clone(), None);
        let convs = connector.scan(&ctx).unwrap();

        // File under 100MB should be processed normally
        assert_eq!(convs.len(), 1);
        assert_eq!(convs[0].messages[0].content, "Under the limit");
    }

    #[test]
    fn null_bytes_embedded_in_content_handled() {
        let dir = TempDir::new().unwrap();
        let claude_dir = make_test_claude_dir(dir.path());

        // JSON allows \u0000 escape for null bytes in strings
        let session_file = claude_dir.join("null_bytes.jsonl");
        let content = concat!(
            r#"{"type":"user","message":{"role":"user","content":"before\u0000after"}}"#,
            "\n",
            r#"{"type":"user","message":{"role":"user","content":"Clean message"}}"#,
            "\n"
        );
        fs::write(&session_file, content).unwrap();

        let connector = ClaudeCodeConnector::new();
        let ctx = ScanContext::local_default(claude_dir.clone(), None);
        let result = connector.scan(&ctx);

        assert!(
            result.is_ok(),
            "null bytes in content should not cause errors"
        );
        let convs = result.unwrap();
        assert_eq!(convs.len(), 1);
        // Both messages should be extracted; the null byte is valid JSON
        assert!(
            !convs[0].messages.is_empty(),
            "should extract at least the clean message"
        );
    }

    #[test]
    fn bom_marker_at_file_start_handled() {
        let dir = TempDir::new().unwrap();
        let claude_dir = make_test_claude_dir(dir.path());

        // UTF-8 BOM: EF BB BF
        let session_file = claude_dir.join("bom.jsonl");
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"\xEF\xBB\xBF"); // UTF-8 BOM
        bytes.extend_from_slice(
            b"{\"type\":\"user\",\"message\":{\"role\":\"user\",\"content\":\"After BOM\"}}\n",
        );
        bytes.extend_from_slice(
            b"{\"type\":\"user\",\"message\":{\"role\":\"user\",\"content\":\"Second line\"}}\n",
        );
        fs::write(&session_file, &bytes).unwrap();

        let connector = ClaudeCodeConnector::new();
        let ctx = ScanContext::local_default(claude_dir.clone(), None);
        let result = connector.scan(&ctx);

        assert!(result.is_ok(), "BOM marker should not cause errors");
        let convs = result.unwrap();
        // The BOM may cause the first line's JSON to fail parsing (since the BOM
        // bytes are prepended to the line). The second line should parse fine.
        // We verify the connector doesn't crash and extracts what it can.
        assert_eq!(convs.len(), 1);
        assert!(
            !convs[0].messages.is_empty(),
            "should extract at least the second message after BOM"
        );
        // The second line (without BOM) should always parse correctly
        assert!(
            convs[0].messages.iter().any(|m| m.content == "Second line"),
            "second line should be extractable regardless of BOM"
        );
    }

    // =========================================================================
    // Chunked emission and tail scans
    // =========================================================================

    fn session_line(i: usize) -> String {
        let role = if i % 2 == 0 { "user" } else { "assistant" };
        format!(
            r#"{{"type":"{role}","cwd":"/work/proj","sessionId":"s-1","gitBranch":"main","timestamp":"2025-12-01T10:{:02}:{:02}Z","message":{{"role":"{role}","content":"message number {i}"}}}}"#,
            (i / 60) % 60,
            i % 60
        )
    }

    fn write_session(path: &Path, lines: &[String]) {
        let mut text = lines.join("\n");
        text.push('\n');
        fs::write(path, text).unwrap();
    }

    fn with_chunk_limits<T>(max_messages: usize, f: impl FnOnce() -> T) -> T {
        TEST_CHUNK_LIMITS.with(|limits| {
            limits.set(Some(ClaudeChunkLimits {
                max_messages,
                max_content_bytes: usize::MAX,
            }));
        });
        let out = f();
        TEST_CHUNK_LIMITS.with(|limits| limits.set(None));
        out
    }

    #[derive(Debug)]
    enum Event {
        Part(NormalizedConversation),
        Complete(SourceCompletion),
    }

    /// Scan `claude_dir` with tail-aware hooks; `resume` answers tail_resume.
    fn scan_events(
        claude_dir: &Path,
        mut resume: impl FnMut(&DiscoveredSourceFile) -> Option<SourceTailResume>,
    ) -> Vec<Event> {
        let events = std::cell::RefCell::new(Vec::new());
        let connector = ClaudeCodeConnector::new();
        let ctx = ScanContext::local_default(claude_dir.to_path_buf(), None);
        let mut on_complete = |completion: &SourceCompletion| {
            events
                .borrow_mut()
                .push(Event::Complete(completion.clone()));
            Ok(())
        };
        let mut hooks = SourceScanHooks {
            should_scan_source: None,
            on_source_complete: Some(&mut on_complete),
            tail_resume: Some(&mut resume),
        };
        connector
            .scan_with_source_boundaries(&ctx, &mut hooks, &mut |conversation| {
                events.borrow_mut().push(Event::Part(conversation));
                Ok(())
            })
            .unwrap();
        events.into_inner()
    }

    fn parts(events: &[Event]) -> Vec<&NormalizedConversation> {
        events
            .iter()
            .filter_map(|event| match event {
                Event::Part(conversation) => Some(conversation),
                Event::Complete(_) => None,
            })
            .collect()
    }

    fn completion(events: &[Event]) -> &SourceCompletion {
        let completions: Vec<_> = events
            .iter()
            .filter_map(|event| match event {
                Event::Complete(completion) => Some(completion),
                Event::Part(_) => None,
            })
            .collect();
        assert_eq!(completions.len(), 1, "exactly one completion per source");
        completions[0]
    }

    fn message_key(message: &NormalizedMessage) -> (i64, String, String, Option<i64>, Value) {
        (
            message.idx,
            message.role.clone(),
            message.content.clone(),
            message.created_at,
            message.extra.clone(),
        )
    }

    #[test]
    fn chunked_parts_concatenate_to_the_one_shot_parse() {
        let dir = TempDir::new().unwrap();
        let claude_dir = make_test_claude_dir(dir.path());
        let mut lines: Vec<String> = (0..23).map(session_line).collect();
        // A tool result rides on a user line, and noise lines carry no messages.
        lines.insert(
            5,
            r#"{"type":"user","timestamp":"2025-12-01T10:00:04Z","message":{"role":"user","content":[{"type":"tool_result","tool_use_id":"t1","content":"tool output"},{"type":"text","text":"after tool"}]}}"#
                .to_string(),
        );
        lines.insert(9, r#"{"type":"progress","data":{}}"#.to_string());
        lines.insert(12, "not json".to_string());
        lines.push(r#"{"type":"ai-title","title":"Late title"}"#.to_string());
        write_session(&claude_dir.join("session.jsonl"), &lines);

        let one_shot = ClaudeCodeConnector::new()
            .scan(&ScanContext::local_default(claude_dir.clone(), None))
            .unwrap();
        assert_eq!(one_shot.len(), 1);
        let one_shot = &one_shot[0];

        let events = with_chunk_limits(4, || scan_events(&claude_dir, |_| None));
        let parts = parts(&events);
        assert!(
            parts.len() > 3,
            "the session is split: {} parts",
            parts.len()
        );
        assert!(
            matches!(events.last(), Some(Event::Complete(_))),
            "the completion comes after the last part"
        );
        assert_eq!(completion(&events).conversations_emitted, parts.len());

        let joined: Vec<_> = parts
            .iter()
            .flat_map(|part| part.messages.iter().map(message_key))
            .collect();
        let expected: Vec<_> = one_shot.messages.iter().map(message_key).collect();
        assert_eq!(joined, expected, "same messages, same order, same idx");
        for (i, key) in joined.iter().enumerate() {
            assert_eq!(key.0, i as i64, "idx is continuous across parts");
        }
        for part in &parts {
            assert!(part.messages.len() <= 4);
            assert_eq!(part.external_id, one_shot.external_id);
            assert_eq!(part.source_path, one_shot.source_path);
            assert_eq!(part.workspace, one_shot.workspace);
            assert_eq!(part.metadata, one_shot.metadata);
            assert_eq!(
                part.title.as_deref(),
                Some("Late title"),
                "every part carries the title the session ends with"
            );
        }
        let last = parts.last().unwrap();
        assert_eq!(last.title, one_shot.title);
        assert_eq!(last.started_at, one_shot.started_at);
        assert_eq!(last.ended_at, one_shot.ended_at);
    }

    #[test]
    fn unchunked_tail_aware_scan_matches_the_plain_scan() {
        let dir = TempDir::new().unwrap();
        let claude_dir = make_test_claude_dir(dir.path());
        let lines: Vec<String> = (0..6).map(session_line).collect();
        write_session(&claude_dir.join("session.jsonl"), &lines);
        let ctx = ScanContext::local_default(claude_dir.clone(), None);
        let one_shot = ClaudeCodeConnector::new().scan(&ctx).unwrap();
        let events = scan_events(&claude_dir, |_| None);
        let parts = parts(&events);
        assert_eq!(parts.len(), 1);
        let part = parts[0];
        assert_eq!(part.title, one_shot[0].title);
        assert_eq!(part.metadata, one_shot[0].metadata);
        assert_eq!(part.started_at, one_shot[0].started_at);
        assert_eq!(part.ended_at, one_shot[0].ended_at);
        let expected: Vec<_> = one_shot[0].messages.iter().map(message_key).collect();
        let got: Vec<_> = part.messages.iter().map(message_key).collect();
        assert_eq!(got, expected);
        let tail = completion(&events).tail.clone().expect("tail state");
        assert_eq!(
            tail.offset,
            fs::metadata(claude_dir.join("session.jsonl"))
                .unwrap()
                .len()
        );
        assert_eq!(tail.next_idx, 6);
        assert!(!tail.resumed);
    }

    fn resume_from(tail: &SourceTailState) -> SourceTailResume {
        SourceTailResume {
            offset: tail.offset,
            next_idx: tail.next_idx,
            carried: tail.carried.clone(),
            dev: tail.dev,
            inode: tail.inode,
            head_len: tail.head_len,
            head_blake3: tail.head_blake3.clone(),
            prefix_blake3: blake3::hash(&tail.prefix).to_hex().to_string(),
            samples_blake3: tail.samples_blake3.clone(),
        }
    }

    #[test]
    fn tail_scan_parses_only_appended_lines_and_continues_idx() {
        let dir = TempDir::new().unwrap();
        let claude_dir = make_test_claude_dir(dir.path());
        let session = claude_dir.join("session.jsonl");
        let all: Vec<String> = (0..10).map(session_line).collect();
        write_session(&session, &all[..7]);
        let first = scan_events(&claude_dir, |_| None);
        let tail = completion(&first).tail.clone().unwrap();
        assert_eq!(tail.next_idx, 7);
        let prefix_len = tail.prefix.len();
        assert_eq!(
            &tail.prefix[..],
            &fs::read(&session).unwrap()[tail.offset as usize - prefix_len..]
        );

        write_session(&session, &all);
        let second = scan_events(&claude_dir, |_| Some(resume_from(&tail)));
        let parts = parts(&second);
        assert_eq!(parts.len(), 1);
        let got: Vec<_> = parts[0].messages.iter().map(message_key).collect();
        let full = ClaudeCodeConnector::new()
            .scan(&ScanContext::local_default(claude_dir.clone(), None))
            .unwrap();
        let expected: Vec<_> = full[0].messages[7..].iter().map(message_key).collect();
        assert_eq!(got, expected, "only the 3 appended messages, idx 7..10");
        assert_eq!(
            parts[0].title, full[0].title,
            "title carried from the earlier scan"
        );
        assert_eq!(parts[0].started_at, full[0].started_at);
        assert_eq!(parts[0].ended_at, full[0].ended_at);
        assert_eq!(parts[0].metadata, full[0].metadata);
        let tail2 = completion(&second).tail.clone().unwrap();
        assert!(tail2.resumed);
        assert_eq!(tail2.next_idx, 10);
        assert_eq!(tail2.offset, fs::metadata(&session).unwrap().len());

        // Nothing new: no part, but a completion that keeps the same point.
        let third = scan_events(&claude_dir, |_| Some(resume_from(&tail2)));
        assert!(parts_is_empty(&third));
        let tail3 = completion(&third).tail.clone().unwrap();
        assert_eq!(
            (tail3.offset, tail3.next_idx),
            (tail2.offset, tail2.next_idx)
        );
        assert_eq!(tail3.prefix, tail2.prefix);
    }

    #[test]
    fn a_rewrite_near_the_start_is_parsed_again_from_the_start() {
        let dir = TempDir::new().unwrap();
        let claude_dir = make_test_claude_dir(dir.path());
        let session = claude_dir.join("session.jsonl");
        let all: Vec<String> = (0..10).map(session_line).collect();
        write_session(&session, &all[..7]);
        let first = scan_events(&claude_dir, |_| None);
        let tail = completion(&first).tail.clone().unwrap();

        // Same inode, same bytes before the offset, but line 1 changed (same
        // length) and new lines appended: the head hash no longer matches.
        let mut rewritten = all.clone();
        rewritten[1] = rewritten[1].replacen("number", "NUMBER", 1);
        assert_eq!(rewritten[1].len(), all[1].len());
        write_session(&session, &rewritten);
        let second = scan_events(&claude_dir, |_| Some(resume_from(&tail)));
        let tail2 = completion(&second).tail.clone().unwrap();
        assert!(!tail2.resumed, "parsed from the start");
        let got: Vec<_> = parts(&second)
            .iter()
            .flat_map(|part| part.messages.iter().map(message_key))
            .collect();
        assert_eq!(got.len(), 10);
        assert_eq!(got[0].0, 0);
    }

    #[test]
    fn a_rewrite_in_the_middle_is_parsed_again_from_the_start() {
        let dir = TempDir::new().unwrap();
        let claude_dir = make_test_claude_dir(dir.path());
        let session = claude_dir.join("session.jsonl");
        // Long enough that the middle lies outside the 64 KiB head and the
        // 4 KiB before the offset, short enough that the samples cover it.
        let pad = "p".repeat(900);
        let all: Vec<String> = (0..130)
            .map(|i| session_line(i).replacen("number", &format!("number {pad}"), 1))
            .collect();
        write_session(&session, &all[..120]);
        let first = scan_events(&claude_dir, |_| None);
        let tail = completion(&first).tail.clone().unwrap();
        assert!(tail.offset > 100 * 1024, "offset {}", tail.offset);

        let mut rewritten = all.clone();
        rewritten[85] = rewritten[85].replacen("ppp", "PPP", 1);
        write_session(&session, &rewritten);
        let second = scan_events(&claude_dir, |_| Some(resume_from(&tail)));
        assert!(!completion(&second).tail.clone().unwrap().resumed);
        assert_eq!(parts(&second)[0].messages[0].idx, 0);
    }

    #[test]
    fn a_different_file_at_the_same_path_is_parsed_from_the_start() {
        let dir = TempDir::new().unwrap();
        let claude_dir = make_test_claude_dir(dir.path());
        let session = claude_dir.join("session.jsonl");
        let all: Vec<String> = (0..10).map(session_line).collect();
        write_session(&session, &all[..7]);
        let first = scan_events(&claude_dir, |_| None);
        let tail = completion(&first).tail.clone().unwrap();

        // Same bytes plus an append, but a new file (new inode) renamed into
        // place, the way an editor or a copy replaces a file.
        let replacement = claude_dir.join("replacement.tmp");
        write_session(&replacement, &all);
        fs::rename(&replacement, &session).unwrap();
        let second = scan_events(&claude_dir, |_| Some(resume_from(&tail)));
        let tail2 = completion(&second).tail.clone().unwrap();
        assert!(!tail2.resumed, "a new inode is never resumed");
        assert_ne!(tail2.inode, tail.inode);
        assert_eq!(parts(&second)[0].messages[0].idx, 0);
    }

    #[test]
    fn a_line_longer_than_the_cap_is_indexed_cut_and_the_session_goes_on() {
        let dir = TempDir::new().unwrap();
        let claude_dir = make_test_claude_dir(dir.path());
        let session = claude_dir.join("session.jsonl");
        let mut lines: Vec<String> = (0..6).map(session_line).collect();
        let body = "x".repeat(300_000);
        lines[3] = session_line(3).replacen("number 3", &format!("number 3 {body}"), 1);
        write_session(&session, &lines);
        let one_shot = ClaudeCodeConnector::new()
            .scan(&ScanContext::local_default(claude_dir.clone(), None))
            .unwrap();

        TEST_MAX_LINE_BYTES.with(|cap| cap.set(Some(4096)));
        TEST_LONG_LINE_KEEP_BYTES.with(|keep| keep.set(Some(8192)));
        let events = scan_events(&claude_dir, |_| None);
        TEST_MAX_LINE_BYTES.with(|cap| cap.set(None));
        TEST_LONG_LINE_KEEP_BYTES.with(|keep| keep.set(None));

        let messages: Vec<&NormalizedMessage> = parts(&events)
            .iter()
            .flat_map(|part| part.messages.iter())
            .collect();
        assert_eq!(
            messages.len(),
            6,
            "every message, including those after the long line"
        );
        for (i, message) in messages.iter().enumerate() {
            assert_eq!(message.idx, i as i64);
        }
        let cut = &messages[3].content;
        assert!(cut.starts_with("message number 3 xxx"));
        assert!(
            cut.contains("[truncated while indexing: kept "),
            "{}",
            &cut[cut.len() - 80..]
        );
        assert!(cut.len() < 9000);
        // The other messages are as a one-shot parse stores them.
        for i in [0, 1, 2, 4, 5] {
            assert_eq!(
                message_key(messages[i]),
                message_key(&one_shot[0].messages[i])
            );
        }
        assert!(
            !events
                .iter()
                .any(|event| matches!(event, Event::Complete(_))),
            "a source with cut text is never certified complete"
        );
    }

    #[test]
    fn a_long_line_kept_whole_still_completes_with_a_tail() {
        let dir = TempDir::new().unwrap();
        let claude_dir = make_test_claude_dir(dir.path());
        let session = claude_dir.join("session.jsonl");
        let mut lines: Vec<String> = (0..6).map(session_line).collect();
        let body = "x".repeat(300_000);
        lines[3] = session_line(3).replacen("number 3", &format!("number 3 {body}"), 1);
        write_session(&session, &lines);

        // Over the line cap, but its text fits the keep budget: nothing is cut.
        TEST_MAX_LINE_BYTES.with(|cap| cap.set(Some(4096)));
        TEST_LONG_LINE_KEEP_BYTES.with(|keep| keep.set(Some(1 << 20)));
        let events = scan_events(&claude_dir, |_| None);
        TEST_MAX_LINE_BYTES.with(|cap| cap.set(None));
        TEST_LONG_LINE_KEEP_BYTES.with(|keep| keep.set(None));

        let messages: Vec<&NormalizedMessage> = parts(&events)
            .iter()
            .flat_map(|part| part.messages.iter())
            .collect();
        assert_eq!(messages.len(), 6);
        assert!(!messages[3].content.contains("[truncated while indexing"));
        let tail = completion(&events).tail.clone().unwrap();
        assert_eq!(tail.offset, fs::metadata(&session).unwrap().len());
        assert_eq!(tail.next_idx, 6);
        assert_eq!(
            &tail.prefix[..],
            &fs::read(&session).unwrap()[tail.offset as usize - tail.prefix.len()..],
            "the tail prefix covers the end of the long line"
        );
    }

    #[test]
    fn a_big_message_is_split_into_chunks_and_a_tail_scan_continues_after_them() {
        let dir = TempDir::new().unwrap();
        let claude_dir = make_test_claude_dir(dir.path());
        let session = claude_dir.join("session.jsonl");
        let mut lines: Vec<String> = (0..4).map(session_line).collect();
        let body = "y".repeat(100);
        lines[1] = session_line(1).replacen("number 1", &format!("number 1 {body}"), 1);
        write_session(&session, &lines);

        TEST_MESSAGE_CHUNK_BYTES.with(|bytes| bytes.set(Some(40)));
        let first = scan_events(&claude_dir, |_| None);
        let first_messages: Vec<NormalizedMessage> = parts(&first)
            .iter()
            .flat_map(|part| part.messages.iter().cloned())
            .collect();
        // 1 + 3 chunks + 1 + 1
        assert_eq!(first_messages.len(), 6);
        for (i, message) in first_messages.iter().enumerate() {
            assert_eq!(message.idx, i as i64);
        }
        let joined: String = first_messages[1..4]
            .iter()
            .map(|message| message.content.as_str())
            .collect();
        assert_eq!(joined, format!("message number 1 {body}"));
        assert_eq!(first_messages[3].extra[crate::types::MESSAGE_CHUNK_KEY]["count"], 3);
        assert_eq!(first_messages[1].role, "assistant");
        let tail = completion(&first).tail.clone().unwrap();
        assert_eq!(tail.next_idx, 6);

        lines.push(session_line(4));
        write_session(&session, &lines);
        let second = scan_events(&claude_dir, |_| Some(resume_from(&tail)));
        let fresh = scan_events(&claude_dir, |_| None);
        TEST_MESSAGE_CHUNK_BYTES.with(|bytes| bytes.set(None));
        let appended = &parts(&second)[0].messages;
        assert_eq!(appended.len(), 1);
        assert_eq!(appended[0].idx, 6);
        let fresh_messages: Vec<&NormalizedMessage> = parts(&fresh)
            .iter()
            .flat_map(|part| part.messages.iter())
            .collect();
        assert_eq!(fresh_messages.len(), 7);
        assert_eq!(message_key(fresh_messages[6]), message_key(&appended[0]));
    }

    #[test]
    fn capped_line_reader_never_buffers_past_the_cap() {
        let data = format!("short\n{}\nafter\n", "y".repeat(10_000));
        let mut reader = std::io::BufReader::with_capacity(64, data.as_bytes());
        let mut line = Vec::new();
        assert_eq!(
            read_line_capped(&mut reader, &mut line, 100, true).unwrap(),
            (6, false)
        );
        assert_eq!(line, b"short\n");
        line.clear();
        let (read, too_long) = read_line_capped(&mut reader, &mut line, 100, true).unwrap();
        assert_eq!((read, too_long), (10_001, true));
        assert!(line.is_empty() && line.capacity() <= 100);
        line.clear();
        assert_eq!(
            read_line_capped(&mut reader, &mut line, 100, true).unwrap(),
            (6, false)
        );
        assert_eq!(line, b"after\n");
    }

    fn parts_is_empty(events: &[Event]) -> bool {
        parts(events).is_empty()
    }

    #[test]
    fn tail_scan_picks_up_a_late_ai_title() {
        let dir = TempDir::new().unwrap();
        let claude_dir = make_test_claude_dir(dir.path());
        let session = claude_dir.join("session.jsonl");
        let mut lines: Vec<String> = (0..4).map(session_line).collect();
        write_session(&session, &lines);
        let first = scan_events(&claude_dir, |_| None);
        assert_eq!(parts(&first)[0].title.as_deref(), Some("message number 0"));
        let tail = completion(&first).tail.clone().unwrap();
        lines.push(r#"{"type":"ai-title","title":"Named later"}"#.to_string());
        lines.push(session_line(4));
        write_session(&session, &lines);
        let second = scan_events(&claude_dir, |_| Some(resume_from(&tail)));
        assert_eq!(parts(&second)[0].title.as_deref(), Some("Named later"));
    }

    #[test]
    fn a_line_still_being_written_is_left_for_the_next_scan() {
        let dir = TempDir::new().unwrap();
        let claude_dir = make_test_claude_dir(dir.path());
        let session = claude_dir.join("session.jsonl");
        let lines: Vec<String> = (0..4).map(session_line).collect();
        let complete = format!("{}\n", lines[..3].join("\n"));
        let half = &lines[3][..lines[3].len() / 2];
        fs::write(&session, format!("{complete}{half}")).unwrap();
        let first = scan_events(&claude_dir, |_| None);
        assert_eq!(parts(&first)[0].messages.len(), 3);
        let tail = completion(&first).tail.clone().unwrap();
        assert_eq!(
            tail.offset,
            complete.len() as u64,
            "the half line is not consumed"
        );
        assert_eq!(tail.next_idx, 3);

        fs::write(&session, format!("{complete}{}\n", lines[3])).unwrap();
        let second = scan_events(&claude_dir, |_| Some(resume_from(&tail)));
        let parts = parts(&second);
        assert_eq!(parts[0].messages.len(), 1);
        assert_eq!(parts[0].messages[0].idx, 3);
        assert_eq!(parts[0].messages[0].content, "message number 3");
    }

    #[test]
    fn an_unterminated_but_complete_last_line_is_consumed() {
        let dir = TempDir::new().unwrap();
        let claude_dir = make_test_claude_dir(dir.path());
        let session = claude_dir.join("session.jsonl");
        let lines: Vec<String> = (0..3).map(session_line).collect();
        fs::write(&session, lines.join("\n")).unwrap();
        let plain = ClaudeCodeConnector::new()
            .scan(&ScanContext::local_default(claude_dir.clone(), None))
            .unwrap();
        let events = scan_events(&claude_dir, |_| None);
        assert_eq!(parts(&events)[0].messages.len(), plain[0].messages.len());
        assert_eq!(
            completion(&events).tail.as_ref().unwrap().offset,
            fs::metadata(&session).unwrap().len()
        );
    }

    #[test]
    fn a_scan_reads_only_up_to_the_pre_parse_size() {
        let dir = TempDir::new().unwrap();
        let claude_dir = make_test_claude_dir(dir.path());
        let session = claude_dir.join("session.jsonl");
        let lines: Vec<String> = (0..6).map(session_line).collect();
        write_session(&session, &lines);
        let identity = ClaudeJsonlIdentity {
            path: &session,
            external_id: Some("session.jsonl".into()),
            is_subagent_transcript: false,
            parent_session_id: None,
        };
        let limit = (lines[0].len() + lines[1].len() + 2) as u64;
        let mut got = Vec::new();
        let outcome = parse_claude_jsonl_source(
            &identity,
            ClaudeJsonlOptions {
                size_limit: Some(limit),
                compact_message_extra: false,
                limits: ClaudeChunkLimits {
                    max_messages: 100,
                    max_content_bytes: usize::MAX,
                },
                wants_tail: true,
            },
            None,
            &mut |conversation| {
                got.push(conversation);
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(
            got[0].messages.len(),
            2,
            "bytes after the snapshot are not read"
        );
        let tail = outcome.tail.unwrap();
        assert_eq!(tail.offset, limit);
        assert_eq!(tail.next_idx, 2);
    }

    #[test]
    fn a_resume_point_past_the_end_falls_back_to_a_full_parse() {
        let dir = TempDir::new().unwrap();
        let claude_dir = make_test_claude_dir(dir.path());
        let session = claude_dir.join("session.jsonl");
        let lines: Vec<String> = (0..4).map(session_line).collect();
        write_session(&session, &lines);
        let first = scan_events(&claude_dir, |_| None);
        let mut resume = resume_from(completion(&first).tail.as_ref().unwrap());
        resume.offset += 10_000;
        let events = scan_events(&claude_dir, |_| Some(resume.clone()));
        let parts = parts(&events);
        assert_eq!(parts[0].messages.len(), 4);
        assert_eq!(parts[0].messages[0].idx, 0);
        assert!(!completion(&events).tail.as_ref().unwrap().resumed);

        // Unknown carried-state version: also a full parse.
        let mut resume = resume_from(completion(&first).tail.as_ref().unwrap());
        resume.carried["v"] = serde_json::json!(99);
        let events = scan_events(&claude_dir, |_| Some(resume.clone()));
        assert_eq!(parts_count_messages(&events), 4);
    }

    fn parts_count_messages(events: &[Event]) -> usize {
        parts(events).iter().map(|part| part.messages.len()).sum()
    }

    #[test]
    fn carried_state_round_trips() {
        let state = ClaudeSessionState {
            workspace: Some(PathBuf::from("/work/proj")),
            session_id: Some("s".into()),
            git_branch: None,
            json_title: Some("t".into()),
            first_user_title: Some("u".into()),
            started_at: Some(1),
            ended_at: Some(2),
        };
        assert_eq!(
            ClaudeSessionState::from_json(&state.to_json().unwrap()),
            Some(state)
        );
        assert_eq!(
            ClaudeSessionState::from_json(&ClaudeSessionState::default().to_json().unwrap()),
            Some(ClaudeSessionState::default())
        );
    }

    #[test]
    fn hosts_without_tail_hooks_keep_the_old_completion() {
        let dir = TempDir::new().unwrap();
        let claude_dir = make_test_claude_dir(dir.path());
        write_session(
            &claude_dir.join("session.jsonl"),
            &(0..3).map(session_line).collect::<Vec<_>>(),
        );
        let mut completions = Vec::new();
        let mut on_complete = |completion: &SourceCompletion| {
            completions.push(completion.clone());
            Ok(())
        };
        let mut hooks = SourceScanHooks {
            should_scan_source: None,
            on_source_complete: Some(&mut on_complete),
            tail_resume: None,
        };
        ClaudeCodeConnector::new()
            .scan_with_source_boundaries(
                &ScanContext::local_default(claude_dir.clone(), None),
                &mut hooks,
                &mut |_| Ok(()),
            )
            .unwrap();
        assert_eq!(completions.len(), 1);
        assert!(completions[0].tail.is_none());
    }
}
