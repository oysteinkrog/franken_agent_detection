//! Normalized types for representing agent conversations.
//!
//! These are the lingua franca types that ALL connectors produce.
//! Any tool (not just cass) can use these types to work with agent session data.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// High-level detection status for a connector.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectionResult {
    pub detected: bool,
    pub evidence: Vec<String>,
    pub root_paths: Vec<PathBuf>,
}

impl DetectionResult {
    #[must_use]
    pub const fn not_found() -> Self {
        Self {
            detected: false,
            evidence: Vec::new(),
            root_paths: Vec::new(),
        }
    }
}

/// Normalized conversation emitted by connectors.
#[cfg(feature = "connectors")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NormalizedConversation {
    pub agent_slug: String,
    pub external_id: Option<String>,
    pub title: Option<String>,
    pub workspace: Option<PathBuf>,
    pub source_path: PathBuf,
    pub started_at: Option<i64>,
    pub ended_at: Option<i64>,
    pub metadata: serde_json::Value,
    pub messages: Vec<NormalizedMessage>,
}

#[cfg(feature = "connectors")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NormalizedMessage {
    pub idx: i64,
    pub role: String,
    pub author: Option<String>,
    pub created_at: Option<i64>,
    pub content: String,
    pub extra: serde_json::Value,
    pub snippets: Vec<NormalizedSnippet>,
    /// Structured tool/skill invocations extracted from this message.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub invocations: Vec<NormalizedInvocation>,
}

/// A single tool or skill invocation within a message.
#[cfg(feature = "connectors")]
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct NormalizedInvocation {
    /// Classification: `"tool"` or `"skill"`.
    pub kind: String,
    /// Canonical searchable name (e.g. `"github-prs"`, `"Read"`, `"bash"`).
    /// For wrapper tools like Amp's `skill("github-prs")`, this is the
    /// unwrapped semantic name, not `"skill"`.
    pub name: String,
    /// Original tool name when different from `name` (e.g. `"skill"` for
    /// Amp skill wrapper calls).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub raw_name: Option<String>,
    /// Provider-assigned call ID, useful for joining with tool results.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub call_id: Option<String>,
    /// Raw input/arguments passed to the tool.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub arguments: Option<serde_json::Value>,
}

#[cfg(feature = "connectors")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NormalizedSnippet {
    pub file_path: Option<PathBuf>,
    pub start_line: Option<i64>,
    pub end_line: Option<i64>,
    pub language: Option<String>,
    pub snippet_text: Option<String>,
}

/// Re-assign sequential indices to messages starting from 0.
/// Use this after filtering or sorting messages to ensure idx values are contiguous.
#[cfg(feature = "connectors")]
#[inline]
pub fn reindex_messages(messages: &mut [NormalizedMessage]) {
    for (i, msg) in messages.iter_mut().enumerate() {
        msg.idx = i64::try_from(i).unwrap_or(i64::MAX);
    }
}

/// Largest message text kept in one message. A longer message is split into
/// ordered chunk messages by [`chunk_message`], so no text is ever cut.
#[cfg(feature = "connectors")]
pub const MESSAGE_CHUNK_BYTES: usize = 4 * 1024 * 1024;

/// Key in `extra` that marks a chunk of a split message.
#[cfg(feature = "connectors")]
pub const MESSAGE_CHUNK_KEY: &str = "cass_chunk";

/// Split a message whose text is over [`MESSAGE_CHUNK_BYTES`] into ordered
/// chunk messages. See [`chunk_message_at`].
#[cfg(feature = "connectors")]
pub fn chunk_message(message: NormalizedMessage) -> Vec<NormalizedMessage> {
    chunk_message_at(message, MESSAGE_CHUNK_BYTES)
}

/// Split a message whose text is over `chunk_bytes` into ordered chunks of at
/// most `chunk_bytes` each, cut only at UTF-8 character boundaries. The chunks'
/// texts joined in order give the original text exactly.
///
/// Every chunk keeps the role, author, time and idx; call this before idx is
/// assigned, so the chunks get consecutive idx. Chunk 0 keeps the original
/// `extra`, invocations and snippets; later chunks carry only the chunk
/// record, so a large raw `extra` is not copied. The chunk record, under
/// [`MESSAGE_CHUNK_KEY`] in `extra`, holds `group` (a hash of the whole text,
/// the same for every chunk of one message), `ordinal`, `count`, the half-open
/// byte range `byte_start..byte_end` in the whole text, and `total_bytes`.
/// When the original `extra` is neither an object nor null, chunk 0 keeps it
/// unchanged and has no chunk record.
#[cfg(feature = "connectors")]
pub fn chunk_message_at(
    mut message: NormalizedMessage,
    chunk_bytes: usize,
) -> Vec<NormalizedMessage> {
    let chunk_bytes = chunk_bytes.max(4);
    if message.content.len() <= chunk_bytes {
        return vec![message];
    }
    let content = std::mem::take(&mut message.content);
    let total = content.len();
    let mut bounds = Vec::new();
    let mut start = 0;
    while start < total {
        let mut end = (start + chunk_bytes).min(total);
        while !content.is_char_boundary(end) {
            end -= 1;
        }
        bounds.push((start, end));
        start = end;
    }
    let hash = blake3::hash(content.as_bytes()).to_hex();
    let group = &hash.as_str()[..16];
    let count = bounds.len();
    let record = |ordinal: usize, (byte_start, byte_end): (usize, usize)| {
        serde_json::json!({
            "group": group,
            "ordinal": ordinal,
            "count": count,
            "byte_start": byte_start,
            "byte_end": byte_end,
            "total_bytes": total,
        })
    };
    let mut chunks = Vec::with_capacity(count);
    for (ordinal, &(byte_start, byte_end)) in bounds.iter().enumerate().skip(1) {
        let mut extra = serde_json::Map::new();
        extra.insert(
            MESSAGE_CHUNK_KEY.to_string(),
            record(ordinal, (byte_start, byte_end)),
        );
        chunks.push(NormalizedMessage {
            idx: message.idx,
            role: message.role.clone(),
            author: message.author.clone(),
            created_at: message.created_at,
            content: content[byte_start..byte_end].to_string(),
            extra: serde_json::Value::Object(extra),
            snippets: Vec::new(),
            invocations: Vec::new(),
        });
    }
    let first = record(0, bounds[0]);
    match &mut message.extra {
        serde_json::Value::Object(map) => {
            map.insert(MESSAGE_CHUNK_KEY.to_string(), first);
        }
        serde_json::Value::Null => {
            let mut map = serde_json::Map::new();
            map.insert(MESSAGE_CHUNK_KEY.to_string(), first);
            message.extra = serde_json::Value::Object(map);
        }
        _ => {}
    }
    let mut content = content;
    content.truncate(bounds[0].1);
    content.shrink_to_fit();
    message.content = content;
    chunks.insert(0, message);
    chunks
}

// -------------------------------------------------------------------------
// Scan & provenance types (feature-gated behind `connectors`)
// -------------------------------------------------------------------------

/// The default source ID for local conversations.
#[cfg(feature = "connectors")]
pub const LOCAL_SOURCE_ID: &str = "local";

/// The kind/type of a source.
#[cfg(feature = "connectors")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum SourceKind {
    /// Local machine (default).
    #[default]
    Local,
    /// Remote machine via SSH.
    Ssh,
}

#[cfg(feature = "connectors")]
impl SourceKind {
    /// Returns true if this is a remote source kind.
    #[must_use]
    pub const fn is_remote(&self) -> bool {
        !matches!(self, Self::Local)
    }

    /// Get the string representation.
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Local => "local",
            Self::Ssh => "ssh",
        }
    }

    /// Parse from string.
    #[must_use]
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "local" => Some(Self::Local),
            "ssh" => Some(Self::Ssh),
            _ => None,
        }
    }
}

#[cfg(feature = "connectors")]
impl std::fmt::Display for SourceKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Per-conversation provenance metadata.
#[cfg(feature = "connectors")]
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Origin {
    /// References Source.id.
    pub source_id: String,
    /// Denormalized source kind for convenience.
    pub kind: SourceKind,
    /// Display host label (may differ from source's `host_label`).
    pub host: Option<String>,
}

#[cfg(feature = "connectors")]
impl Origin {
    /// Create an origin for local conversations.
    #[must_use]
    pub fn local() -> Self {
        Self {
            source_id: LOCAL_SOURCE_ID.to_string(),
            kind: SourceKind::Local,
            host: None,
        }
    }

    /// Create an origin for remote conversations.
    #[must_use]
    pub fn remote(source_id: impl Into<String>) -> Self {
        let id = source_id.into();
        Self {
            source_id: id.clone(),
            kind: SourceKind::Ssh,
            host: Some(id),
        }
    }

    /// Create an origin for remote conversations with explicit host label.
    #[must_use]
    pub fn remote_with_host(source_id: impl Into<String>, host: impl Into<String>) -> Self {
        Self {
            source_id: source_id.into(),
            kind: SourceKind::Ssh,
            host: Some(host.into()),
        }
    }

    /// Check if this origin is from a remote source.
    #[must_use]
    pub const fn is_remote(&self) -> bool {
        self.kind.is_remote()
    }

    /// Check if this origin is local.
    ///
    /// Like [`Origin::is_remote`], this is kind-based: any origin whose
    /// `kind` is [`SourceKind::Local`] is local, including named local
    /// sources whose `source_id` differs from [`LOCAL_SOURCE_ID`]. Callers
    /// that specifically need the default local source should compare
    /// `source_id == LOCAL_SOURCE_ID` explicitly.
    #[must_use]
    pub const fn is_local(&self) -> bool {
        matches!(self.kind, SourceKind::Local)
    }

    /// Get a display label for this origin.
    #[must_use]
    pub fn display_label(&self) -> String {
        match (&self.host, &self.kind) {
            (Some(host), SourceKind::Ssh) => format!("{host} (remote)"),
            (Some(host), SourceKind::Local) => host.clone(),
            (None, SourceKind::Local) => "local".to_string(),
            (None, SourceKind::Ssh) => format!("{} (remote)", self.source_id),
        }
    }

    /// Get a short display label (just the identifier, no suffix).
    #[must_use]
    pub fn short_label(&self) -> &str {
        self.host.as_deref().unwrap_or(&self.source_id)
    }
}

#[cfg(feature = "connectors")]
impl Default for Origin {
    fn default() -> Self {
        Self::local()
    }
}

#[cfg(feature = "connectors")]
pub(crate) fn agent_name_matches_filter(allowed: &str, actual: &str) -> bool {
    let normalize = |value: &str| value.trim().to_ascii_lowercase().replace('-', "_");
    normalize(allowed) == normalize(actual)
}

/// A single path mapping rule for rewriting paths.
#[cfg(feature = "connectors")]
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PathMapping {
    /// Source path prefix to match.
    pub from: String,
    /// Target path prefix to replace with.
    pub to: String,
    /// Optional: only apply this mapping for specific agents.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub agents: Option<Vec<String>>,
}

#[cfg(feature = "connectors")]
impl PathMapping {
    /// Create a new path mapping.
    #[must_use]
    pub fn new(from: impl Into<String>, to: impl Into<String>) -> Self {
        Self {
            from: from.into(),
            to: to.into(),
            agents: None,
        }
    }

    /// Create a new path mapping with agent filter.
    #[must_use]
    pub fn with_agents(
        from: impl Into<String>,
        to: impl Into<String>,
        agents: Vec<String>,
    ) -> Self {
        Self {
            from: from.into(),
            to: to.into(),
            agents: Some(agents),
        }
    }

    /// Apply this mapping to a path if it matches.
    #[must_use]
    pub fn apply(&self, path: &str) -> Option<String> {
        if path == self.from {
            return Some(self.to.clone());
        }

        if !path.starts_with(&self.from) {
            return None;
        }

        let rest = &path[self.from.len()..];
        let from_ends_with_sep = self.from.ends_with('/') || self.from.ends_with('\\');
        let rest_starts_with_sep = rest.starts_with(['/', '\\']);

        // Boundary check: the match must land on a path separator on at
        // least one side, otherwise we'd be matching a filename substring
        // (e.g. `from="/a"` vs `path="/ab"`).
        if !from_ends_with_sep && !rest_starts_with_sep {
            return None;
        }

        let to_ends_with_sep = self.to.ends_with('/') || self.to.ends_with('\\');

        // Emit exactly one separator at the splice, regardless of which
        // side carried it:
        //
        //   to_sep   rest_sep   action
        //   ------   --------   -------------------------------------
        //   true     true       drop the leading sep from `rest`
        //   true     false      concatenate — `to` already has the sep
        //   false    true       concatenate — `rest` carries the sep
        //   false    false      insert a sep; `from` must have had one
        //                       (by the boundary check), so use the same
        //                       flavor it used.
        //
        // Without this, `from="/a"`, `to="/b/"`, `path="/a/file"` used to
        // produce `"/b//file"` because `to` ended with `/` and `rest`
        // began with `/`, and the old branch-shape only inserted a
        // separator when `from` had ended with one. Double-separator
        // output canonicalizes identically under POSIX but breaks
        // string-based path-equality checks in downstream consumers.
        let rewritten = match (to_ends_with_sep, rest_starts_with_sep) {
            (true, true) => format!("{}{}", self.to, &rest[1..]),
            (true, false) | (false, true) => format!("{}{}", self.to, rest),
            (false, false) => {
                let sep = if self.from.ends_with('\\') { '\\' } else { '/' };
                format!("{}{}{}", self.to, sep, rest)
            }
        };
        Some(rewritten)
    }

    /// Check if this mapping applies to a given agent.
    #[must_use]
    pub fn applies_to_agent(&self, agent: Option<&str>) -> bool {
        match (&self.agents, agent) {
            (None, _) => true,
            (Some(_), None) => false,
            (Some(agents), Some(a)) => agents
                .iter()
                .any(|allowed| agent_name_matches_filter(allowed, a)),
        }
    }
}

/// Platform hint for choosing default paths.
#[cfg(feature = "connectors")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Platform {
    Macos,
    Linux,
    Windows,
}

#[cfg(feature = "connectors")]
impl std::fmt::Display for Platform {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Macos => write!(f, "macos"),
            Self::Linux => write!(f, "linux"),
            Self::Windows => write!(f, "windows"),
        }
    }
}

#[cfg(all(test, feature = "connectors"))]
mod tests {
    use super::*;

    fn text_message(content: &str, extra: serde_json::Value) -> NormalizedMessage {
        NormalizedMessage {
            idx: 7,
            role: "tool".to_string(),
            author: Some("a".to_string()),
            created_at: Some(1_700_000_000_000),
            content: content.to_string(),
            extra,
            snippets: Vec::new(),
            invocations: vec![NormalizedInvocation {
                kind: "tool".to_string(),
                name: "Read".to_string(),
                raw_name: None,
                call_id: Some("c1".to_string()),
                arguments: None,
            }],
        }
    }

    #[test]
    fn a_short_message_is_not_chunked() {
        let message = text_message("short", serde_json::json!({"k": 1}));
        let chunks = chunk_message_at(message, 16);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].content, "short");
        assert_eq!(chunks[0].extra, serde_json::json!({"k": 1}));
    }

    #[test]
    fn chunks_rebuild_the_text_exactly_and_never_split_a_character() {
        // 'é' is 2 bytes and '€' is 3, so 10-byte chunks must back off.
        let text = "aé€".repeat(40);
        let chunks = chunk_message_at(text_message(&text, serde_json::json!({"k": 1})), 10);
        assert!(chunks.len() > 1);
        let joined: String = chunks.iter().map(|chunk| chunk.content.as_str()).collect();
        assert_eq!(joined, text);
        let group = chunks[0].extra[MESSAGE_CHUNK_KEY]["group"].clone();
        let mut next = 0;
        for (ordinal, chunk) in chunks.iter().enumerate() {
            assert!(chunk.content.len() <= 10);
            assert_eq!((chunk.idx, chunk.role.as_str()), (7, "tool"));
            assert_eq!(chunk.created_at, Some(1_700_000_000_000));
            let record = &chunk.extra[MESSAGE_CHUNK_KEY];
            assert_eq!(record["group"], group);
            assert_eq!(record["ordinal"], ordinal);
            assert_eq!(record["count"], chunks.len());
            assert_eq!(record["total_bytes"], text.len());
            assert_eq!(record["byte_start"], next);
            next += chunk.content.len();
            assert_eq!(record["byte_end"], next);
            assert_eq!(
                &text[record["byte_start"].as_u64().unwrap() as usize..next],
                chunk.content
            );
        }
        // Chunk 0 keeps the original extra and the invocations; the rest carry
        // only the chunk record.
        assert_eq!(chunks[0].extra["k"], 1);
        assert_eq!(chunks[0].invocations.len(), 1);
        for chunk in &chunks[1..] {
            assert_eq!(chunk.extra.as_object().unwrap().len(), 1);
            assert!(chunk.invocations.is_empty());
        }
    }

    #[test]
    fn chunk_records_on_null_and_scalar_extra() {
        let chunks = chunk_message_at(text_message(&"x".repeat(30), serde_json::Value::Null), 10);
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].extra[MESSAGE_CHUNK_KEY]["ordinal"], 0);
        let chunks = chunk_message_at(text_message(&"x".repeat(30), serde_json::json!("raw")), 10);
        assert_eq!(chunks[0].extra, serde_json::json!("raw"));
        assert_eq!(chunks[1].extra[MESSAGE_CHUNK_KEY]["ordinal"], 1);
    }

    #[test]
    fn default_local_origin_is_local_not_remote() {
        let origin = Origin::local();
        assert!(origin.is_local());
        assert!(!origin.is_remote());
        assert_eq!(origin.source_id, LOCAL_SOURCE_ID);
    }

    #[test]
    fn named_local_kind_origin_is_local_not_remote() {
        // Regression for issue #23: a named local source (source_id !=
        // LOCAL_SOURCE_ID but kind == Local) must classify as local, matching
        // the kind-based semantics of `is_remote`.
        let origin = Origin {
            source_id: "backup-local".to_string(),
            kind: SourceKind::Local,
            host: None,
        };
        assert!(origin.is_local());
        assert!(!origin.is_remote());
    }

    #[test]
    fn remote_origin_is_remote_not_local() {
        let origin = Origin::remote("workstation");
        assert!(origin.is_remote());
        assert!(!origin.is_local());
    }
}
