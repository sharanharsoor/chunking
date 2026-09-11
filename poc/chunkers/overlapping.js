/* overlapping_window. Sliding window on characters, words, or simple_v1 sentences. */
(function (root) {
  function wordSpans(text) {
    var re = /\S+/g;
    var spans = [];
    var m;
    while ((m = re.exec(text)) !== null) {
      spans.push({ start: m.index, end: m.index + m[0].length });
    }
    return spans;
  }

  function sentenceSpans(text) {
    var sentences;
    if (typeof root.splitSentencesSimpleV1 === "function") {
      sentences = root.splitSentencesSimpleV1(text);
    } else {
      sentences = text.split(/(?<=[.!?])\s+/).filter(Boolean);
    }
    var spans = [];
    var pos = 0;
    for (var i = 0; i < sentences.length; i++) {
      var s = sentences[i];
      var idx = text.indexOf(s, pos);
      if (idx < 0) idx = text.indexOf(s);
      if (idx < 0) {
        spans.push({ start: pos, end: pos + s.length });
        pos += s.length;
      } else {
        spans.push({ start: idx, end: idx + s.length });
        pos = idx + s.length;
      }
    }
    return spans;
  }

  function slide(spans, text, windowSize, stepSize, idPrefix, extra) {
    var chunks = [];
    if (!spans.length) return chunks;
    var i = 0;
    var n = 0;
    while (i < spans.length) {
      var endIdx = Math.min(spans.length, i + windowSize);
      chunks.push(
        root.chunkFromUtf16(
          idPrefix + n,
          text,
          spans[i].start,
          spans[endIdx - 1].end,
          extra
        )
      );
      n += 1;
      i += stepSize;
    }
    return chunks;
  }

  function chunkOverlappingWindow(text, params, enc) {
    var windowSize = Math.max(1, Number(params.window_size) || 500);
    var stepSize = Math.max(1, Number(params.step_size) || 250);
    var unit = params.window_unit || "words";
    var maxTokens = Number(params.max_tokens) || 0;
    var overlapTokens = Number(params.overlap_tokens);
    if (unit === "tokens" && !maxTokens) {
      maxTokens = windowSize;
      overlapTokens = Math.max(0, windowSize - stepSize);
      unit = "tokens";
    }
    if (maxTokens || unit === "tokens") {
      if (!enc || typeof enc.encode !== "function" || typeof enc.decode !== "function") {
        throw new Error("TOKENIZER");
      }
      if (!Number.isFinite(overlapTokens) || overlapTokens < 0) overlapTokens = 0;
      if (overlapTokens >= maxTokens) overlapTokens = Math.max(0, maxTokens - 1);
      var ids = Array.prototype.slice.call(enc.encode(text));
      var windows = root.tokenWindows(ids.length, maxTokens, overlapTokens);
      var chunks = [];
      var hint = 0;
      var chars = Array.from(text);
      for (var w = 0; w < windows.length; w++) {
        var slice = ids.slice(windows[w][0], windows[w][1]);
        var content = enc.decode(slice);
        var start = 0;
        var end = chars.length;
        if (content) {
          var found = text.indexOf(content, hint);
          if (found < 0) found = text.indexOf(content);
          if (found >= 0) {
            start = Array.from(text.slice(0, found)).length;
            end = start + Array.from(content).length;
            hint = found + (overlapTokens ? 1 : content.length);
          }
        }
        chunks.push({
          id: "overlapping_window_" + w,
          content: content,
          start: start,
          end: end,
          size: content.length,
          offset_unit: "char",
          metadata: {
            chunker_used: "overlapping_window",
            source: params.source || "paste",
            window_unit: "tokens",
            token_count: slice.length,
            start_token_index: windows[w][0],
          },
        });
      }
      return chunks;
    }
    if (stepSize >= windowSize) {
      throw new Error("step_size must be less than window_size");
    }
    var extra = {
      chunker_used: "overlapping_window",
      source: params.source || "paste",
      window_unit: unit,
    };
    if (unit === "characters") {
      var charChunks = [];
      var startPos = 0;
      var n = 0;
      var all = Array.from(text);
      while (startPos < all.length) {
        var endPos = Math.min(all.length, startPos + windowSize);
        charChunks.push(
          root.chunkFromScalar("overlapping_window_" + n, text, startPos, endPos, extra)
        );
        n += 1;
        startPos += stepSize;
      }
      return charChunks;
    }
    var spans = unit === "sentences" ? sentenceSpans(text) : wordSpans(text);
    return slide(spans, text, windowSize, stepSize, "overlapping_window_", extra);
  }

  root.chunkOverlappingWindow = chunkOverlappingWindow;
})(typeof self !== "undefined" ? self : typeof global !== "undefined" ? global : window);
