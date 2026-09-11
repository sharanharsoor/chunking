/* Shared Unicode-scalar offsets for in-tab chunkers. */
(function (root) {
  function utf16ToScalar(str, utf16Index) {
    if (utf16Index <= 0) return 0;
    if (utf16Index >= str.length) return Array.from(str).length;
    return Array.from(str.slice(0, utf16Index)).length;
  }

  function chunkFromUtf16(id, text, utf16Start, utf16End, extra) {
    utf16Start = Math.max(0, utf16Start);
    utf16End = Math.min(text.length, Math.max(utf16Start, utf16End));
    var content = text.slice(utf16Start, utf16End);
    return {
      id: id,
      content: content,
      start: utf16ToScalar(text, utf16Start),
      end: utf16ToScalar(text, utf16End),
      size: content.length,
      offset_unit: "char",
      metadata: extra || {},
    };
  }

  function chunkFromScalar(id, text, start, end, extra) {
    var chars = Array.from(text);
    start = Math.max(0, start);
    end = Math.min(chars.length, Math.max(start, end));
    var content = chars.slice(start, end).join("");
    return {
      id: id,
      content: content,
      start: start,
      end: end,
      size: content.length,
      offset_unit: "char",
      metadata: extra || {},
    };
  }

  function trimUtf16Span(text, start, end) {
    while (start < end && /\s/.test(text.charAt(start))) start += 1;
    while (end > start && /\s/.test(text.charAt(end - 1))) end -= 1;
    return { start: start, end: end };
  }

  function chunksFromSpans(idPrefix, text, spans, extra) {
    if (!spans.length) {
      if (!text) return [];
      return [chunkFromUtf16(idPrefix + "0", text, 0, text.length, extra)];
    }
    return spans.map(function (s, i) {
      return chunkFromUtf16(idPrefix + i, text, s.start, s.end, extra);
    });
  }

  root.utf16ToScalar = utf16ToScalar;
  root.chunkFromUtf16 = chunkFromUtf16;
  root.chunkFromScalar = chunkFromScalar;
  root.trimUtf16Span = trimUtf16Span;
  root.chunksFromSpans = chunksFromSpans;
})(typeof self !== "undefined" ? self : typeof global !== "undefined" ? global : window);
