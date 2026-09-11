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

  function chunkOverlappingWindow(text, params) {
    var windowSize = Math.max(1, Number(params.window_size) || 500);
    var stepSize = Math.max(1, Number(params.step_size) || 250);
    var unit = params.window_unit || "words";
    if (stepSize >= windowSize) {
      throw new Error("step_size must be less than window_size");
    }
    var extra = {
      chunker_used: "overlapping_window",
      source: params.source || "paste",
      window_unit: unit,
    };
    if (unit === "characters") {
      var chars = Array.from(text);
      var chunks = [];
      var startPos = 0;
      var n = 0;
      while (startPos < chars.length) {
        var endPos = Math.min(chars.length, startPos + windowSize);
        chunks.push(
          root.chunkFromScalar("overlapping_window_" + n, text, startPos, endPos, extra)
        );
        n += 1;
        startPos += stepSize;
      }
      return chunks;
    }
    var spans = unit === "sentences" ? sentenceSpans(text) : wordSpans(text);
    return slide(spans, text, windowSize, stepSize, "overlapping_window_", extra);
  }

  root.chunkOverlappingWindow = chunkOverlappingWindow;
})(typeof self !== "undefined" ? self : typeof global !== "undefined" ? global : window);
