/* regex_custom. Split at each match; the match starts the next chunk. */
(function (root) {
  function chunkRegexCustom(text, params) {
    var pattern = params.pattern == null ? "\\n\\n" : String(params.pattern);
    var multiline = params.multiline !== false;
    if (!text) return [];
    if (!pattern) {
      return [root.chunkFromUtf16("regex_custom_0", text, 0, text.length, {
        chunker_used: "regex_custom",
        source: params.source || "paste",
        pattern: pattern,
        multiline: multiline,
      })];
    }
    var flags = multiline ? "gm" : "g";
    var rx;
    try {
      rx = new RegExp(pattern, flags);
    } catch (err) {
      throw new Error("INVALID_REGEX");
    }
    var starts = [];
    var m;
    var guard = 0;
    while ((m = rx.exec(text)) !== null) {
      starts.push(m.index);
      if (!m[0] || m[0].length === 0) rx.lastIndex += 1;
      guard += 1;
      if (guard > 200000) break;
    }
    if (!starts.length) starts = [0];
    if (starts[0] !== 0) starts.unshift(0);
    var uniq = [];
    var i;
    for (i = 0; i < starts.length; i++) {
      if (!uniq.length || starts[i] > uniq[uniq.length - 1]) uniq.push(starts[i]);
    }
    var chunks = [];
    for (i = 0; i < uniq.length; i++) {
      var utfStart = uniq[i];
      var utfEnd = i + 1 < uniq.length ? uniq[i + 1] : text.length;
      if (utfEnd <= utfStart) continue;
      chunks.push(root.chunkFromUtf16("regex_custom_" + chunks.length, text, utfStart, utfEnd, {
        chunker_used: "regex_custom",
        source: params.source || "paste",
        pattern: pattern,
        multiline: multiline,
      }));
    }
    return chunks;
  }

  root.chunkRegexCustom = chunkRegexCustom;
})(typeof self !== "undefined" ? self : typeof global !== "undefined" ? global : window);
