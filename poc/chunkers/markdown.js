/* markdown_chunker, chunk_by=headers. Split on ATX headers up to header_level. */
(function (root) {
  function maskFences(text) {
    return text.replace(/```[\s\S]*?```/g, function (block) {
      return block.replace(/[^\n]/g, " ");
    });
  }

  function chunkMarkdown(text, params) {
    var level = Math.max(1, Math.min(6, Number(params.header_level) || 2));
    var masked = maskFences(text);
    var re = /^(#{1,6})[ \t]+(.+)$/gm;
    var headers = [];
    var m;
    while ((m = re.exec(masked)) !== null) {
      if (m[1].length <= level) {
        headers.push({ start: m.index, level: m[1].length, title: m[2].trim() });
      }
    }
    var extra = { chunker_used: "markdown_chunker", source: params.source || "paste", chunk_by: "headers" };
    if (!headers.length) {
      if (!text) return [];
      return [root.chunkFromUtf16("markdown_chunker_0", text, 0, text.length, extra)];
    }
    var chunks = [];
    for (var i = 0; i < headers.length; i++) {
      var start = i === 0 ? 0 : headers[i].start;
      var end = i + 1 < headers.length ? headers[i + 1].start : text.length;
      var span = root.trimUtf16Span(text, start, end);
      if (span.end <= span.start) continue;
      chunks.push(
        root.chunkFromUtf16("markdown_chunker_" + chunks.length, text, span.start, span.end, extra)
      );
    }
    return chunks;
  }

  root.chunkMarkdown = chunkMarkdown;
})(typeof self !== "undefined" ? self : typeof global !== "undefined" ? global : window);
