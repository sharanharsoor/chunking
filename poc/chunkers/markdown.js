/* markdown_chunker, chunk_by=headers. Split on ATX headers up to header_level. */
(function (root) {
  function maskFences(text) {
    return text.replace(/```[\s\S]*?```|~~~[\s\S]*?~~~/g, function (block) {
      return block.replace(/[^\n]/g, " ");
    });
  }

  function collectHeaders(text) {
    var masked = maskFences(text);
    var re = /^(#{1,6})[ \t]+(.+)$/gm;
    var headers = [];
    var m;
    while ((m = re.exec(masked)) !== null) {
      headers.push({ start: m.index, level: m[1].length, title: m[2].trim() });
    }
    return headers;
  }

  function headerPath(all, at) {
    var stack = [];
    for (var i = 0; i < all.length; i++) {
      var h = all[i];
      if (h.start > at.start) break;
      while (stack.length && stack[stack.length - 1].level >= h.level) stack.pop();
      stack.push(h);
    }
    return stack.map(function (h) { return h.title; });
  }

  function lineSpans(text) {
    var lines = [];
    var pos = 0;
    var parts = text.split(/(\n)/);
    var buf = "";
    var start = 0;
    function flush(end) {
      if (buf.length) lines.push({ start: start, end: end, line: buf });
      buf = "";
      start = end;
    }
    for (var i = 0; i < parts.length; i++) {
      var p = parts[i];
      if (p === "\n") {
        buf += p;
        pos += 1;
        flush(pos);
      } else if (p) {
        if (!buf) start = pos;
        buf += p;
        pos += p.length;
      }
    }
    if (buf) flush(pos);
    if (!lines.length && text) lines.push({ start: 0, end: text.length, line: text });
    return lines;
  }

  function atomicSpans(text) {
    var lines = lineSpans(text);
    var spans = [];
    var i = 0;
    function isFence(line) {
      var s = line.replace(/^\s+/, "");
      return s.indexOf("```") === 0 || s.indexOf("~~~") === 0;
    }
    function fenceDelim(line) {
      var s = line.replace(/^\s+/, "");
      return s.slice(0, 3);
    }
    function isGfm(line) {
      return /^\s*\|.*\|\s*$/.test(line.replace(/\n$/, ""));
    }
    while (i < lines.length) {
      var start = lines[i].start;
      var line = lines[i].line;
      if (isFence(line)) {
        var delim = fenceDelim(line);
        var j = i + 1;
        while (j < lines.length && lines[j].line.replace(/^\s+/, "").indexOf(delim) !== 0) j += 1;
        if (j < lines.length) j += 1;
        spans.push({ start: start, end: lines[Math.max(i, j - 1)].end, kind: "code" });
        i = j;
        continue;
      }
      if (/<table\b/i.test(line)) {
        var t = i;
        while (t < lines.length && !/<\/table>/i.test(lines[t].line)) t += 1;
        if (t < lines.length) t += 1;
        spans.push({ start: start, end: lines[Math.max(i, t - 1)].end, kind: "html_table" });
        i = Math.max(t, i + 1);
        continue;
      }
      if (isGfm(line)) {
        var g = i + 1;
        while (g < lines.length && isGfm(lines[g].line)) g += 1;
        spans.push({ start: start, end: lines[g - 1].end, kind: "gfm_table" });
        i = g;
        continue;
      }
      var k = i + 1;
      while (k < lines.length) {
        var nxt = lines[k].line;
        if (!nxt.trim()) {
          k += 1;
          break;
        }
        if (isFence(nxt) || /<table\b/i.test(nxt) || isGfm(nxt)) break;
        k += 1;
      }
      spans.push({ start: start, end: lines[k - 1].end, kind: "text" });
      i = k;
    }
    return spans;
  }

  function packAtoms(text, maxTokens, enc) {
    var atoms = atomicSpans(text);
    if (!atoms.length) return [{ start: 0, end: text.length }];
    var packed = [];
    var cur = [];
    var tok = 0;
    function flush() {
      if (!cur.length) return;
      packed.push({ start: cur[0].start, end: cur[cur.length - 1].end });
      cur = [];
      tok = 0;
    }
    for (var a = 0; a < atoms.length; a++) {
      var piece = text.slice(atoms[a].start, atoms[a].end);
      var ptok = enc.encode(piece).length;
      if (cur.length && tok + ptok > maxTokens) flush();
      cur.push(atoms[a]);
      tok += ptok;
      if (cur.length === 1 && ptok > maxTokens) flush();
    }
    flush();
    return packed;
  }

  function decorate(chunk, path, params) {
    var extra = {
      chunker_used: "markdown_chunker",
      source: params.source || "paste",
      chunk_by: "headers",
    };
    if (params.breadcrumb !== false && path.length) {
      extra.header_path = path.slice();
      extra.breadcrumb = path.join(" > ");
    }
    if (params.contextualize && extra.breadcrumb) {
      extra.raw_content = chunk.content;
      chunk.content = "[" + extra.breadcrumb + "]\n\n" + chunk.content;
      chunk.size = chunk.content.length;
    }
    chunk.metadata = extra;
    return chunk;
  }

  function chunkMarkdown(text, params, enc) {
    var level = Math.max(1, Math.min(6, Number(params.header_level) || 2));
    var maxTokens = Number(params.max_tokens) || 0;
    if (maxTokens && (!enc || typeof enc.encode !== "function")) throw new Error("TOKENIZER");
    var all = collectHeaders(text);
    var headers = all.filter(function (h) { return h.level <= level; });
    function emit(id, utfStart, utfEnd, path) {
      var span = root.trimUtf16Span(text, utfStart, utfEnd);
      if (span.end <= span.start) return null;
      var body = text.slice(span.start, span.end);
      if (maxTokens) {
        var pieces = packAtoms(body, maxTokens, enc);
        var out = [];
        for (var p = 0; p < pieces.length; p++) {
          var inner = root.trimUtf16Span(body, pieces[p].start, pieces[p].end);
          if (inner.end <= inner.start) continue;
          var absStart = span.start + inner.start;
          var absEnd = span.start + inner.end;
          out.push(decorate(
            root.chunkFromUtf16(id + (out.length ? "_" + out.length : ""), text, absStart, absEnd, {}),
            path,
            params
          ));
        }
        return out;
      }
      return [decorate(root.chunkFromUtf16(id, text, span.start, span.end, {}), path, params)];
    }
    if (!headers.length) {
      if (!text) return [];
      var none = emit("markdown_chunker_0", 0, text.length, []);
      return none || [];
    }
    var chunks = [];
    for (var i = 0; i < headers.length; i++) {
      var start = i === 0 ? 0 : headers[i].start;
      var end = i + 1 < headers.length ? headers[i + 1].start : text.length;
      var path = headerPath(all, headers[i]);
      var made = emit("markdown_chunker_" + chunks.length, start, end, path);
      if (made) Array.prototype.push.apply(chunks, made);
    }
    return chunks;
  }

  root.chunkMarkdown = chunkMarkdown;
})(typeof self !== "undefined" ? self : typeof global !== "undefined" ? global : window);
