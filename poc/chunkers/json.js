/* json_chunker, chunk_by=objects. Arrays and JSONL group objects_per_chunk; a single object is one chunk. */
(function (root) {
  function skipWs(text, i) {
    while (i < text.length && /[ \t\r\n]/.test(text[i])) i += 1;
    return i;
  }

  function skipString(text, i) {
    i += 1;
    while (i < text.length) {
      if (text[i] === "\\") {
        i += 2;
        continue;
      }
      if (text[i] === '"') return i + 1;
      i += 1;
    }
    return i;
  }

  function skipValue(text, i) {
    i = skipWs(text, i);
    var c = text[i];
    if (c === '"') return skipString(text, i);
    if (c === "{" || c === "[") {
      var close = c === "{" ? "}" : "]";
      var depth = 1;
      i += 1;
      while (i < text.length && depth) {
        if (text[i] === '"') {
          i = skipString(text, i);
          continue;
        }
        if (text[i] === c) depth += 1;
        else if (text[i] === close) depth -= 1;
        i += 1;
      }
      return i;
    }
    while (i < text.length && /[^\s,\]\}]/.test(text[i])) i += 1;
    return i;
  }

  function arraySpans(text) {
    var i = skipWs(text, 0);
    if (text[i] !== "[") return null;
    i += 1;
    var spans = [];
    while (i < text.length) {
      i = skipWs(text, i);
      if (text[i] === "]") break;
      if (text[i] === ",") {
        i += 1;
        continue;
      }
      var start = i;
      i = skipValue(text, i);
      if (i <= start) break;
      spans.push({ start: start, end: i });
    }
    return spans;
  }

  function jsonlSpans(text) {
    var spans = [];
    var start = 0;
    for (var i = 0; i <= text.length; i++) {
      if (i !== text.length && text[i] !== "\n") continue;
      var end = i;
      if (end > start && text[end - 1] === "\r") end -= 1;
      var line = text.slice(start, end).trim();
      if (line) {
        try {
          JSON.parse(line);
          var trimmed = root.trimUtf16Span(text, start, end);
          spans.push(trimmed);
        } catch (err) {
          return null;
        }
      }
      start = i + 1;
    }
    return spans.length ? spans : null;
  }

  function groupSpans(spans, text, per, extra) {
    var chunks = [];
    for (var i = 0; i < spans.length; i += per) {
      var group = spans.slice(i, i + per);
      var meta = Object.assign({}, extra, {
        json_object_count: group.length,
        json_start_index: i,
        json_end_index: i + group.length - 1,
      });
      chunks.push(
        root.chunkFromUtf16(
          "json_chunker_" + chunks.length,
          text,
          group[0].start,
          group[group.length - 1].end,
          meta
        )
      );
    }
    return chunks;
  }

  function chunkJson(text, params) {
    var per = Math.max(1, Number(params.objects_per_chunk) || 100);
    var extra = { chunker_used: "json_chunker", source: params.source || "paste", chunk_by: "objects" };
    var raw = text.replace(/^\uFEFF/, "");
    if (!raw.trim()) return [];
    try {
      var data = JSON.parse(raw);
      if (Array.isArray(data)) {
        var spans = arraySpans(text);
        if (spans && spans.length) return groupSpans(spans, text, per, extra);
      }
      return [root.chunkFromUtf16("json_chunker_0", text, 0, text.length, extra)];
    } catch (err) {
      var lines = jsonlSpans(text);
      if (lines) return groupSpans(lines, text, per, extra);
      return [root.chunkFromUtf16("json_chunker_0", text, 0, text.length, extra)];
    }
  }

  root.chunkJson = chunkJson;
})(typeof self !== "undefined" ? self : typeof global !== "undefined" ? global : window);
