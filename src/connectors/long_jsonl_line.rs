//! Streaming parse of one JSONL line that is too long to buffer.
//!
//! The Claude Code parser buffers each line whole and hands it to
//! `serde_json`, which costs several times the line size. A line longer than
//! the buffer cap is read here instead, one byte at a time from the reader,
//! into a `serde_json::Value` of the same shape. Strings are kept only up to a
//! shared byte budget (strings up to 4 KiB, such as keys and timestamps, are
//! always kept); a string cut short ends with a marker that says how long it
//! was. Memory stays near the budget whatever the line length, and
//! every field (type, timestamp, role) is still there, so the line indexes as
//! a normal message with truncated text.

use serde_json::{Map, Number, Value};
use std::io::BufRead;

/// Deepest nesting accepted before the line is treated as invalid. The same
/// limit as `serde_json`, which parses every shorter line.
const MAX_DEPTH: usize = 128;

/// Values (objects, arrays, scalars) kept per line. Later ones are read and
/// dropped, so a line of millions of tiny values stays bounded.
const MAX_KEPT_VALUES: usize = 1_000_000;

/// Longest number token kept as a number; a longer one is kept as text.
const MAX_NUMBER_BYTES: usize = 4096;

/// Nesting depth, from the line's top value, inside which keys and short
/// strings may still use the structure pool when both allowances are spent.
/// It covers `message.content[].text` and the like.
const POOL_DEPTH: usize = 4;

/// Size of that pool.
const POOL_BYTES: usize = 1024 * 1024;

/// Whether `token` is a JSON number: -?(0|[1-9][0-9]*)(.[0-9]+)?([eE][+-]?[0-9]+)?
fn is_json_number(token: &[u8]) -> bool {
    let mut i = 0;
    let digits = |i: &mut usize| {
        let start = *i;
        while *i < token.len() && token[*i].is_ascii_digit() {
            *i += 1;
        }
        *i > start
    };
    if token.get(i) == Some(&b'-') {
        i += 1;
    }
    match token.get(i) {
        Some(b'0') => i += 1,
        Some(b'1'..=b'9') => {
            digits(&mut i);
        }
        _ => return false,
    }
    if token.get(i) == Some(&b'.') {
        i += 1;
        if !digits(&mut i) {
            return false;
        }
    }
    if matches!(token.get(i), Some(b'e' | b'E')) {
        i += 1;
        if matches!(token.get(i), Some(b'+' | b'-')) {
            i += 1;
        }
        if !digits(&mut i) {
            return false;
        }
    }
    i == token.len()
}

/// Streaming check of a number token too long to hold: the same grammar,
/// one byte at a time.
#[derive(Clone, Copy, PartialEq)]
enum NumberState {
    Sign,
    Zero,
    Int,
    Dot,
    Frac,
    Exp,
    ExpSign,
    ExpDigits,
    Bad,
}

impl NumberState {
    fn start(first: u8) -> Self {
        match first {
            b'-' => Self::Sign,
            b'0' => Self::Zero,
            b'1'..=b'9' => Self::Int,
            _ => Self::Bad,
        }
    }

    fn step(self, byte: u8) -> Self {
        match (self, byte) {
            (Self::Sign, b'0') => Self::Zero,
            (Self::Sign, b'1'..=b'9') | (Self::Int, b'0'..=b'9') => Self::Int,
            (Self::Zero | Self::Int, b'.') => Self::Dot,
            (Self::Dot | Self::Frac, b'0'..=b'9') => Self::Frac,
            (Self::Zero | Self::Int | Self::Frac, b'e' | b'E') => Self::Exp,
            (Self::Exp, b'+' | b'-') => Self::ExpSign,
            (Self::Exp | Self::ExpSign | Self::ExpDigits, b'0'..=b'9') => Self::ExpDigits,
            _ => Self::Bad,
        }
    }

    fn complete(self) -> bool {
        matches!(self, Self::Zero | Self::Int | Self::Frac | Self::ExpDigits)
    }
}

/// Strings up to this size are always kept whole (keys, types, timestamps),
/// whatever is left of the budget.
const SMALL_STRING_BYTES: usize = 4096;

/// How a streamed line ended.
#[derive(Debug, PartialEq)]
pub(crate) enum LongLine {
    /// A complete JSON value, with the bytes consumed (including the
    /// newline, when there was one) and how many strings were cut.
    Parsed {
        value: Value,
        consumed: u64,
        truncated_strings: usize,
        /// Values read and dropped past the per-line limits.
        dropped_values: usize,
    },
    /// Not valid JSON. The rest of the line was consumed (`consumed` bytes
    /// in all), so parsing can go on with the next line.
    Invalid { consumed: u64 },
    /// The reader ended before the value did: a line still being written.
    Incomplete,
}

/// Text appended to a string that was cut to fit the budget.
pub(crate) fn truncation_marker(kept: usize, total: u64) -> String {
    format!("\n[truncated while indexing: kept {kept} of {total} bytes]")
}

struct Source<'a, R: BufRead> {
    reader: &'a mut R,
    consumed: u64,
    /// The last bytes consumed, for the tail prefix the caller keeps.
    tail: &'a mut Vec<u8>,
    tail_cap: usize,
    last: Option<u8>,
}

enum Stop {
    Invalid,
    Eof,
    Io,
}

impl<R: BufRead> Source<'_, R> {
    fn peek(&mut self) -> Result<Option<u8>, Stop> {
        let buf = self.reader.fill_buf().map_err(|_| Stop::Io)?;
        Ok(buf.first().copied())
    }

    fn next(&mut self) -> Result<u8, Stop> {
        let byte = self.peek()?.ok_or(Stop::Eof)?;
        self.reader.consume(1);
        self.consumed += 1;
        self.last = Some(byte);
        if self.tail_cap > 0 {
            if self.tail.len() >= self.tail_cap * 2 {
                self.tail.drain(..self.tail.len() - self.tail_cap);
            }
            self.tail.push(byte);
        }
        Ok(byte)
    }

    fn skip_ws(&mut self) -> Result<(), Stop> {
        while let Some(byte) = self.peek()? {
            match byte {
                b' ' | b'\t' | b'\r' => {
                    self.next()?;
                }
                _ => break,
            }
        }
        Ok(())
    }

    fn expect_literal(&mut self, rest: &[u8]) -> Result<(), Stop> {
        for &want in rest {
            if self.next()? != want {
                return Err(Stop::Invalid);
            }
        }
        Ok(())
    }
}

struct Parser {
    /// What is left of the budget for long string values.
    budget_left: usize,
    /// What is left for keys and short strings once `budget_left` is spent
    /// (the same size again). When both are spent, strings are kept empty
    /// and object entries are dropped.
    extra_left: usize,
    values_left: usize,
    /// Set while a value is read only to be dropped.
    discard: bool,
    truncated_strings: usize,
    dropped_values: usize,
    /// Set for a top-level key and its value: kept up to 4 KiB even when
    /// both allowances are spent, so type, timestamp and role survive.
    reserve: bool,
    /// Whether the last string read was cut.
    last_string_cut: bool,
    /// Cut keys seen so far; each gets a numbered suffix.
    cut_keys: usize,
    /// Depth of the string being read, for the structure pool.
    depth: usize,
    /// What is left of the structure pool (see `POOL_DEPTH`).
    pool_left: usize,
    /// The budget each allowance starts with.
    keep_bytes: usize,
}

impl Parser {
    fn value<R: BufRead>(&mut self, src: &mut Source<'_, R>, depth: usize) -> Result<Value, Stop> {
        if depth > MAX_DEPTH {
            return Err(Stop::Invalid);
        }
        src.skip_ws()?;
        self.depth = depth;
        let reserve = std::mem::take(&mut self.reserve);
        match src.next()? {
            b'{' => self.object(src, depth),
            b'[' => self.array(src, depth),
            b'"' => {
                self.reserve = reserve;
                let text = self.string(src, false);
                self.reserve = false;
                text.map(Value::String)
            }
            b't' => src.expect_literal(b"rue").map(|()| Value::Bool(true)),
            b'f' => src.expect_literal(b"alse").map(|()| Value::Bool(false)),
            b'n' => src.expect_literal(b"ull").map(|()| Value::Null),
            first @ (b'-' | b'0'..=b'9') => self.number(src, first, reserve),
            _ => Err(Stop::Invalid),
        }
    }

    /// Read one value and drop it.
    fn skip_value<R: BufRead>(
        &mut self,
        src: &mut Source<'_, R>,
        depth: usize,
    ) -> Result<(), Stop> {
        let was = std::mem::replace(&mut self.discard, true);
        let result = self.value(src, depth).map(drop);
        self.discard = was;
        self.dropped_values += 1;
        result
    }

    /// Whether the next container entry can be kept.
    fn take_value_slot(&mut self) -> bool {
        if self.discard || self.values_left == 0 {
            return false;
        }
        self.values_left -= 1;
        true
    }

    fn object<R: BufRead>(&mut self, src: &mut Source<'_, R>, depth: usize) -> Result<Value, Stop> {
        let mut map = Map::new();
        src.skip_ws()?;
        if src.peek()? == Some(b'}') {
            src.next()?;
            return Ok(Value::Object(map));
        }
        loop {
            src.skip_ws()?;
            if src.next()? != b'"' {
                return Err(Stop::Invalid);
            }
            // Top-level fields (type, timestamp, message) are kept even
            // when the value limit is spent, up to a few hundred of them.
            let top_level = depth == 0 && !self.discard && map.len() < 256;
            let room = self.extra_left > 0 || (depth < POOL_DEPTH && self.pool_left > 0);
            let keep = top_level || (self.take_value_slot() && room);
            let key = if keep {
                self.depth = depth + 1;
                self.reserve = top_level;
                let key = self.string(src, true);
                self.reserve = false;
                let key = key?;
                if self.last_string_cut {
                    // A cut key can repeat; give it its own free name so it
                    // never replaces an earlier value. Whole keys keep the
                    // usual JSON rule: the last value wins.
                    loop {
                        self.cut_keys += 1;
                        let name = format!("{key} #{}", self.cut_keys);
                        if !map.contains_key(&name) {
                            break name;
                        }
                    }
                } else {
                    key
                }
            } else {
                let was = std::mem::replace(&mut self.discard, true);
                let key = self.string(src, true);
                self.discard = was;
                key?
            };
            src.skip_ws()?;
            if src.next()? != b':' {
                return Err(Stop::Invalid);
            }
            if keep && top_level && key == "message" {
                // The message is what gets indexed: it gets allowances of its
                // own, so fields before it (such as a huge toolUseResult)
                // can never spend them.
                let saved = (
                    self.budget_left,
                    self.extra_left,
                    self.pool_left,
                    self.values_left,
                );
                self.budget_left = self.keep_bytes;
                self.extra_left = self.keep_bytes;
                self.pool_left = POOL_BYTES;
                self.values_left = MAX_KEPT_VALUES;
                self.reserve = true;
                let value = self.value(src, depth + 1);
                self.reserve = false;
                (
                    self.budget_left,
                    self.extra_left,
                    self.pool_left,
                    self.values_left,
                ) = saved;
                map.insert(key, value?);
            } else if keep {
                self.reserve = top_level;
                let value = self.value(src, depth + 1);
                self.reserve = false;
                map.insert(key, value?);
            } else {
                self.skip_value(src, depth + 1)?;
            }
            src.skip_ws()?;
            match src.next()? {
                b',' => {}
                b'}' => return Ok(Value::Object(map)),
                _ => return Err(Stop::Invalid),
            }
        }
    }

    fn array<R: BufRead>(&mut self, src: &mut Source<'_, R>, depth: usize) -> Result<Value, Stop> {
        let mut items = Vec::new();
        src.skip_ws()?;
        if src.peek()? == Some(b']') {
            src.next()?;
            return Ok(Value::Array(items));
        }
        loop {
            if self.take_value_slot() {
                items.push(self.value(src, depth + 1)?);
            } else {
                self.skip_value(src, depth + 1)?;
            }
            src.skip_ws()?;
            match src.next()? {
                b',' => {}
                b']' => return Ok(Value::Array(items)),
                _ => return Err(Stop::Invalid),
            }
        }
    }

    fn number<R: BufRead>(
        &mut self,
        src: &mut Source<'_, R>,
        first: u8,
        reserve: bool,
    ) -> Result<Value, Stop> {
        let mut text = vec![first];
        let mut total = 1u64;
        let mut state = NumberState::start(first);
        while let Some(byte) = src.peek()? {
            if !matches!(byte, b'0'..=b'9' | b'-' | b'+' | b'.' | b'e' | b'E') {
                break;
            }
            src.next()?;
            total += 1;
            state = state.step(byte);
            if text.len() < MAX_NUMBER_BYTES {
                text.push(byte);
            }
        }
        // A malformed number makes the line invalid, as with serde_json,
        // also inside a value that is being dropped.
        if !state.complete() {
            return Err(Stop::Invalid);
        }
        if self.discard {
            return Ok(Value::Null);
        }
        if total as usize <= MAX_NUMBER_BYTES {
            debug_assert!(is_json_number(&text));
            let text = String::from_utf8(text).map_err(|_| Stop::Invalid)?;
            if let Ok(number) = serde_json::from_str::<Number>(&text) {
                return Ok(Value::Number(number));
            }
            // Valid but out of range here (such as 1e400): keep its text.
            return Ok(self.number_text(text, total, reserve));
        }
        // Longer than any real number: keep its first 4 KiB as text.
        let text = String::from_utf8(text).map_err(|_| Stop::Invalid)?;
        Ok(self.number_text(text, total, reserve))
    }

    /// A number kept as text, paid from the extra allowance like a short
    /// string; null when that is spent.
    fn number_text(&mut self, text: String, total: u64, reserve: bool) -> Value {
        let kept = text.len();
        let text = if total > kept as u64 {
            self.truncated_strings += 1;
            text + &truncation_marker(kept, total)
        } else {
            text
        };
        if reserve {
            return Value::String(text);
        }
        if text.len() <= self.extra_left {
            self.extra_left -= text.len();
            return Value::String(text);
        }
        if self.depth <= POOL_DEPTH && text.len() <= self.pool_left {
            self.pool_left -= text.len();
            return Value::String(text);
        }
        self.dropped_values += 1;
        Value::Null
    }

    fn hex4<R: BufRead>(src: &mut Source<'_, R>) -> Result<u16, Stop> {
        let mut unit = 0u16;
        for _ in 0..4 {
            let digit = char::from(src.next()?).to_digit(16).ok_or(Stop::Invalid)?;
            unit = unit * 16 + u16::try_from(digit).map_err(|_| Stop::Invalid)?;
        }
        Ok(unit)
    }

    /// The rest of a string after its opening quote. Bytes past what the
    /// string may keep are read and dropped. A long value string uses the
    /// main budget; keys and short strings may also use the extra allowance.
    fn string<R: BufRead>(&mut self, src: &mut Source<'_, R>, key: bool) -> Result<String, Stop> {
        let small = if self.reserve {
            SMALL_STRING_BYTES
        } else if self.extra_left > 0 {
            SMALL_STRING_BYTES.min(self.extra_left)
        } else if self.depth <= POOL_DEPTH {
            SMALL_STRING_BYTES.min(self.pool_left)
        } else {
            0
        };
        let allowed = if self.discard {
            0
        } else if key {
            small
        } else {
            self.budget_left.max(small)
        };
        let mut kept: Vec<u8> = Vec::new();
        let mut total: u64 = 0;
        let mut cut = false;
        let mut push = |bytes: &[u8]| {
            total += bytes.len() as u64;
            if !cut && kept.len() + bytes.len() <= allowed {
                kept.extend_from_slice(bytes);
            } else {
                cut = true;
            }
        };
        loop {
            let byte = src.next()?;
            match byte {
                b'"' => break,
                b'\\' => {
                    let decoded = match src.next()? {
                        b'"' => '"',
                        b'\\' => '\\',
                        b'/' => '/',
                        b'b' => '\u{8}',
                        b'f' => '\u{c}',
                        b'n' => '\n',
                        b'r' => '\r',
                        b't' => '\t',
                        b'u' => {
                            let unit = Self::hex4(src)?;
                            if (0xD800..0xDC00).contains(&unit) {
                                // A high surrogate needs its low half.
                                if src.next()? != b'\\' || src.next()? != b'u' {
                                    return Err(Stop::Invalid);
                                }
                                let low = Self::hex4(src)?;
                                if !(0xDC00..0xE000).contains(&low) {
                                    return Err(Stop::Invalid);
                                }
                                let code = 0x10000
                                    + ((u32::from(unit) - 0xD800) << 10)
                                    + (u32::from(low) - 0xDC00);
                                char::from_u32(code).ok_or(Stop::Invalid)?
                            } else if (0xDC00..0xE000).contains(&unit) {
                                return Err(Stop::Invalid);
                            } else {
                                char::from_u32(u32::from(unit)).ok_or(Stop::Invalid)?
                            }
                        }
                        _ => return Err(Stop::Invalid),
                    };
                    let mut utf8 = [0u8; 4];
                    push(decoded.encode_utf8(&mut utf8).as_bytes());
                }
                // A raw control character (including a newline) is not
                // allowed inside a JSON string.
                0x00..=0x1f => return Err(Stop::Invalid),
                _ => push(&[byte]),
            }
        }
        self.last_string_cut = cut;
        if self.discard {
            return Ok(String::new());
        }
        // Charge the main budget first (value strings only), then the extra.
        let from_budget = if key {
            0
        } else {
            kept.len().min(self.budget_left)
        };
        self.budget_left -= from_budget;
        let rest = kept.len() - from_budget;
        let from_extra = rest.min(self.extra_left);
        self.extra_left -= from_extra;
        self.pool_left = self.pool_left.saturating_sub(rest - from_extra);
        if !cut {
            return String::from_utf8(kept).map_err(|_| Stop::Invalid);
        }
        self.truncated_strings += 1;
        // Cut on a character boundary.
        let valid = match std::str::from_utf8(&kept) {
            Ok(_) => kept.len(),
            Err(error) => error.valid_up_to(),
        };
        kept.truncate(valid);
        let mut text = String::from_utf8(kept).map_err(|_| Stop::Invalid)?;
        let marker = truncation_marker(text.len(), total);
        if self.reserve || marker.len() <= self.extra_left {
            self.extra_left = self.extra_left.saturating_sub(marker.len());
            text.push_str(&marker);
        }
        Ok(text)
    }
}

/// Parse one line from `reader`, which is positioned at its first byte.
/// String bytes beyond `keep_bytes` (shared by all strings of the line) are
/// dropped. `tail` receives the last `tail_cap` (or more) consumed bytes.
pub(crate) fn parse_long_line<R: BufRead>(
    reader: &mut R,
    keep_bytes: usize,
    tail: &mut Vec<u8>,
    tail_cap: usize,
) -> LongLine {
    let mut src = Source {
        reader,
        consumed: 0,
        tail,
        tail_cap,
        last: None,
    };
    let mut parser = Parser {
        budget_left: keep_bytes,
        extra_left: keep_bytes,
        values_left: MAX_KEPT_VALUES,
        discard: false,
        truncated_strings: 0,
        dropped_values: 0,
        reserve: false,
        last_string_cut: false,
        cut_keys: 0,
        depth: 0,
        pool_left: POOL_BYTES,
        keep_bytes,
    };
    // A byte order mark may start the line, as in the buffered path.
    let parsed = (|| {
        if src.peek()? == Some(0xEF) {
            src.expect_literal(&[0xEF, 0xBB, 0xBF])?;
        }
        let value = parser.value(&mut src, 0)?;
        src.skip_ws()?;
        match src.peek()? {
            None => Ok(value),
            Some(b'\n') => {
                src.next()?;
                Ok(value)
            }
            Some(_) => Err(Stop::Invalid),
        }
    })();
    match parsed {
        Ok(value) => LongLine::Parsed {
            value,
            consumed: src.consumed,
            truncated_strings: parser.truncated_strings,
            dropped_values: parser.dropped_values,
        },
        Err(Stop::Eof | Stop::Io) => LongLine::Incomplete,
        Err(Stop::Invalid) => {
            // Skip to the end of the line; a line with no newline yet may
            // still be being written.
            if src.last == Some(b'\n') {
                return LongLine::Invalid {
                    consumed: src.consumed,
                };
            }
            loop {
                match src.next() {
                    Ok(b'\n') => {
                        return LongLine::Invalid {
                            consumed: src.consumed,
                        };
                    }
                    Ok(_) => {}
                    Err(_) => return LongLine::Incomplete,
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(text: &str, keep: usize) -> LongLine {
        let mut tail = Vec::new();
        parse_long_line(&mut text.as_bytes(), keep, &mut tail, 16)
    }

    #[test]
    fn matches_serde_json_when_nothing_is_cut() {
        let lines = [
            r#"{"type":"user","timestamp":"2025-12-01T10:00:00Z","message":{"role":"user","content":"plain"}}"#,
            r#"{"a":[1,-2.5,3e10,true,false,null,{"b":[]},{}],"s":"tab\tquote\"back\\slash\/ \u00e9 \ud83d\ude00 \u0000 é 😀"}"#,
            "  {\"x\" : [ 1 , 2 ] }  ",
            r#"[]"#,
            r#""just a string""#,
            r#"{"nested":[[[[["deep"]]]]],"n":-0,"big":12345678901234567890}"#,
        ];
        for line in lines {
            let expected: Value = serde_json::from_str(line).unwrap();
            match parse(&format!("{line}\n"), usize::MAX) {
                LongLine::Parsed {
                    value,
                    consumed,
                    truncated_strings,
                    dropped_values,
                } => {
                    assert_eq!(dropped_values, 0);
                    assert_eq!(value, expected, "{line}");
                    assert_eq!(consumed, line.len() as u64 + 1);
                    assert_eq!(truncated_strings, 0);
                }
                other => panic!("{line}: {other:?}"),
            }
        }
    }

    #[test]
    fn long_strings_are_cut_on_a_char_boundary_with_a_marker() {
        let body = "é".repeat(5000);
        let line = format!(r#"{{"type":"user","message":{{"content":"{body}"}},"timestamp":"t"}}"#);
        let LongLine::Parsed {
            value,
            truncated_strings,
            ..
        } = parse(&line, 6001)
        else {
            panic!("not parsed");
        };
        assert_eq!(truncated_strings, 1);
        // The message has its own 6,001 bytes; the content keeps 3,000 whole
        // two-byte characters.
        let content = value["message"]["content"].as_str().unwrap();
        assert!(content.starts_with(&"é".repeat(3000)));
        assert!(
            content.ends_with(&truncation_marker(6000, 10_000)),
            "{content}"
        );
        // Short strings after the long one are still kept whole.
        assert_eq!(value["timestamp"], "t");
        assert_eq!(value["type"], "user");
    }

    #[test]
    fn many_tiny_values_are_bounded_and_top_level_fields_survive() {
        let nulls = vec!["null"; MAX_KEPT_VALUES + 10].join(",");
        let line =
            format!(r#"{{"type":"user","message":{{"content":[{nulls}]}},"timestamp":"t"}}"#);
        let LongLine::Parsed {
            value,
            dropped_values,
            ..
        } = parse(&line, 1 << 20)
        else {
            panic!("not parsed");
        };
        assert!(dropped_values >= 10, "{dropped_values}");
        let kept = value["message"]["content"].as_array().unwrap().len();
        assert!(kept <= MAX_KEPT_VALUES);
        assert_eq!(value["timestamp"], "t", "a top-level field after the limit");
    }

    #[test]
    fn cut_keys_never_replace_each_other_and_whole_keys_keep_the_last_value() {
        let a = "k".repeat(SMALL_STRING_BYTES + 10);
        let line = format!(r#"{{"{a}":1,"{a}":2,"type":"old","type":"new"}}"#);
        let LongLine::Parsed { value, .. } = parse(&line, 1 << 20) else {
            panic!("not parsed");
        };
        let map = value.as_object().unwrap();
        assert_eq!(map.len(), 3, "{:?}", map.keys().collect::<Vec<_>>());
        let values: Vec<_> = map.values().cloned().collect();
        assert!(values.contains(&Value::from(1)) && values.contains(&Value::from(2)));
        assert_eq!(value["type"], "new", "as serde_json: the last value wins");
        // Many repeated empty keys finish quickly and keep one entry.
        let empties = vec![r#""":0"#; 200_000].join(",");
        let LongLine::Parsed { value, .. } = parse(&format!("{{{empties}}}"), 1 << 20) else {
            panic!("not parsed");
        };
        assert_eq!(value.as_object().unwrap().len(), 1);
    }

    #[test]
    fn top_level_fields_survive_when_the_allowances_are_spent() {
        let filler = vec![format!(r#""{}""#, "f".repeat(4000)); 50].join(",");
        let line = format!(
            r#"{{"type":"user","message":{{"content":[{filler}]}},"timestamp":"2026-10-01T10:00:00Z"}}"#
        );
        let LongLine::Parsed { value, .. } = parse(&line, 8192) else {
            panic!("not parsed");
        };
        assert_eq!(value["timestamp"], "2026-10-01T10:00:00Z");
        assert_eq!(value["type"], "user");
    }

    #[test]
    fn message_content_survives_when_the_allowances_are_spent() {
        // Enough filler to spend the budget, the extra allowance and the
        // whole structure pool before the message.
        let filler = vec![format!(r#""{}""#, "f".repeat(4000)); 400].join(",");
        let line = format!(
            r#"{{"toolUseResult":[{filler}],"message":{{"role":"user","content":[{{"type":"text","text":"hello"}}]}},"timestamp":"t"}}"#
        );
        let LongLine::Parsed { value, .. } = parse(&line, 8192) else {
            panic!("not parsed");
        };
        assert_eq!(value["message"]["role"], "user");
        assert_eq!(value["message"]["content"][0]["text"], "hello");
        assert_eq!(value["timestamp"], "t");
    }

    #[test]
    fn number_grammar_matches_json() {
        for good in ["0", "-0", "12", "1.5", "1e5", "1E+5", "-1.25e-3", "1e400"] {
            assert!(is_json_number(good.as_bytes()), "{good}");
            let mut state = NumberState::start(good.as_bytes()[0]);
            for &b in &good.as_bytes()[1..] {
                state = state.step(b);
            }
            assert!(state.complete(), "{good}");
        }
        for bad in ["-", "01", "1.", ".5", "1e", "1e+", "1-2", "--1", "1.2.3"] {
            assert!(!is_json_number(bad.as_bytes()), "{bad}");
            let mut state = NumberState::start(bad.as_bytes()[0]);
            for &b in &bad.as_bytes()[1..] {
                state = state.step(b);
            }
            assert!(!state.complete(), "{bad}");
        }
        let LongLine::Parsed { value, .. } = parse("{\"n\":1e400,\"t\":1}\n", 1 << 20) else {
            panic!("an out-of-range number keeps the line");
        };
        assert_eq!(value["n"], "1e400");
        // A malformed number in a dropped value still makes the line invalid.
        let nulls = vec!["null"; MAX_KEPT_VALUES + 2].join(",");
        let line = format!("{{\"a\":[{nulls},1e+]}}\n");
        assert!(matches!(parse(&line, 1 << 20), LongLine::Invalid { .. }));
    }

    #[test]
    fn long_numbers_are_kept_as_text_and_deep_nesting_is_invalid() {
        let zeros = format!("0.{}", "0".repeat(200));
        let LongLine::Parsed { value, .. } = parse(&format!(r#"{{"n":{zeros}}}"#), 1 << 20) else {
            panic!("not parsed");
        };
        assert_eq!(value["n"].as_f64(), Some(0.0));
        let huge = "9".repeat(MAX_NUMBER_BYTES + 5);
        let LongLine::Parsed { value, .. } = parse(&format!(r#"{{"n":{huge},"t":1}}"#), 1 << 20)
        else {
            panic!("not parsed");
        };
        assert!(
            value["n"]
                .as_str()
                .unwrap()
                .contains("[truncated while indexing")
        );
        assert_eq!(value["t"], 1);
        assert!(matches!(
            parse("{\"n\":1e+}\n", 1 << 20),
            LongLine::Invalid { .. }
        ));
        let deep = format!(
            "{}{}\n",
            "[".repeat(MAX_DEPTH + 2),
            "]".repeat(MAX_DEPTH + 2)
        );
        assert!(matches!(parse(&deep, 1 << 20), LongLine::Invalid { .. }));
    }

    #[test]
    fn invalid_lines_are_skipped_to_their_end_and_unfinished_ones_wait() {
        assert_eq!(
            parse("{\"a\":tru}\nnext", usize::MAX),
            LongLine::Invalid { consumed: 10 }
        );
        assert_eq!(parse("{\"a\":\"open", usize::MAX), LongLine::Incomplete);
        assert_eq!(
            parse("{\"a\":1} x\n", usize::MAX),
            LongLine::Invalid { consumed: 10 }
        );
        assert_eq!(
            parse("{\"a\":\"raw\nnewline\"}\n", usize::MAX),
            LongLine::Invalid { consumed: 10 }
        );
    }

    #[test]
    fn memory_stays_near_the_budget_on_a_huge_string() {
        // 64 MiB of string from a reader that never holds it all.
        struct Repeat {
            left: usize,
            stage: u8,
        }
        impl std::io::Read for Repeat {
            fn read(&mut self, out: &mut [u8]) -> std::io::Result<usize> {
                let head = br#"{"type":"user","message":{"content":""#;
                match self.stage {
                    0 => {
                        out[..head.len()].copy_from_slice(head);
                        self.stage = 1;
                        Ok(head.len())
                    }
                    1 if self.left > 0 => {
                        let n = out.len().min(self.left);
                        out[..n].fill(b'x');
                        self.left -= n;
                        Ok(n)
                    }
                    1 => {
                        let end = b"\"}}\n";
                        out[..end.len()].copy_from_slice(end);
                        self.stage = 2;
                        Ok(end.len())
                    }
                    _ => Ok(0),
                }
            }
        }
        let mut reader = std::io::BufReader::with_capacity(
            64 * 1024,
            Repeat {
                left: 64 << 20,
                stage: 0,
            },
        );
        let mut tail = Vec::new();
        let LongLine::Parsed {
            value, consumed, ..
        } = parse_long_line(&mut reader, 1 << 20, &mut tail, 4096)
        else {
            panic!("not parsed");
        };
        let content = value["message"]["content"].as_str().unwrap();
        assert!(content.len() < (1 << 20) + 100);
        assert!(content.ends_with(&truncation_marker(1 << 20, 64 << 20)));
        assert_eq!(consumed, (64u64 << 20) + 37 + 4);
        assert!(tail.len() <= 8192 && tail.ends_with(b"\"}}\n"));
    }
}
