/* fixed_length_word. Word windows with optional overlap. */
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

  function chunkFixedLengthWord(text, params) {
    var n = Math.max(1, Number(params.words_per_chunk) || 100);
    var overlap = Math.max(0, Number(params.overlap_words) || 0);
    if (overlap >= n) overlap = Math.max(0, n - 1);
    var extra = { chunker_used: "fixed_length_word", source: params.source || "paste" };
    var words = wordSpans(text);
    if (!words.length) {
      if (!text) return [];
      return [root.chunkFromUtf16("fixed_length_word_0", text, 0, text.length, extra)];
    }
    var step = Math.max(1, n - overlap);
    var chunks = [];
    var i = 0;
    while (i < words.length) {
      var endIdx = Math.min(words.length, i + n);
      chunks.push(
        root.chunkFromUtf16(
          "fixed_length_word_" + chunks.length,
          text,
          words[i].start,
          words[endIdx - 1].end,
          extra
        )
      );
      if (endIdx >= words.length) break;
      i += step;
    }
    return chunks;
  }

  root.chunkFixedLengthWord = chunkFixedLengthWord;
})(typeof self !== "undefined" ? self : typeof global !== "undefined" ? global : window);
