/* token_based. cl100k_base windows. Encoder is passed in — never chars/4. */
(function (root) {
  function lengthOf(text) {
    return Array.from(text).length;
  }

  function scalarIndexOf(haystack, needle, from) {
    if (!needle) return from || 0;
    var chars = Array.from(haystack);
    var want = Array.from(needle);
    var n = chars.length;
    var m = want.length;
    function scan(start) {
      var i, j, ok;
      for (i = Math.max(0, start); i <= n - m; i++) {
        ok = true;
        for (j = 0; j < m; j++) {
          if (chars[i + j] !== want[j]) {
            ok = false;
            break;
          }
        }
        if (ok) return i;
      }
      return -1;
    }
    var idx = scan(from || 0);
    if (idx >= 0) return idx;
    if (from > 0) return scan(0);
    return -1;
  }

  function asTokenList(tokens) {
    if (!tokens) return [];
    if (typeof tokens.slice === "function") return Array.prototype.slice.call(tokens);
    return Array.from(tokens);
  }

  function chunkTokenBased(text, params, enc) {
    if (!enc || typeof enc.encode !== "function" || typeof enc.decode !== "function") {
      throw new Error("TOKENIZER");
    }
    var n = Math.max(1, Number(params.tokens_per_chunk) || 48);
    var overlap = Number(params.overlap_tokens);
    if (!Number.isFinite(overlap) || overlap < 0) overlap = 0;
    if (overlap >= n) overlap = Math.max(0, n - 1);
    var minTokens = Math.max(1, Number(params.min_chunk_tokens) || 1);
    if (!text) return [];
    var tokens = asTokenList(enc.encode(text));
    var total = tokens.length;
    if (!total) return [];
    var step = Math.max(1, n - overlap);
    var chunks = [];
    var hint = 0;
    var startIdx;
    for (startIdx = 0; startIdx < total; startIdx += step) {
      var endIdx = Math.min(startIdx + n, total);
      var slice = tokens.slice(startIdx, endIdx);
      if (slice.length < minTokens && endIdx < total) continue;
      var content = enc.decode(slice);
      var start = content ? scalarIndexOf(text, content, hint) : hint;
      if (start < 0) start = hint;
      var end = start + lengthOf(content);
      chunks.push({
        id: "token_" + chunks.length,
        content: content,
        start: start,
        end: end,
        size: content.length,
        offset_unit: "char",
        metadata: {
          chunker_used: "token_based",
          source: params.source || "paste",
          token_count: slice.length,
          start_token_index: startIdx,
          end_token_index: endIdx - 1,
          tokenizer_type: "tiktoken",
          tokenizer_model: params.tokenizer_model || "gpt-3.5-turbo",
          tokenizer_encoding: "cl100k_base",
          preserve_word_boundaries: false,
        },
      });
      hint = overlap ? Math.max(start + 1, hint) : end;
      if (endIdx >= total) break;
    }
    return chunks;
  }

  root.chunkTokenBased = chunkTokenBased;
})(typeof self !== "undefined" ? self : typeof global !== "undefined" ? global : window);
