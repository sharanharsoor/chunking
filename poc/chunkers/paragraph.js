/* paragraph_based. Split on blank lines, group max_paragraphs. Offsets stay on the original text. */
(function (root) {
  function paragraphSpans(text) {
    var re = /\r?\n[ \t]*\r?\n/g;
    var spans = [];
    var last = 0;
    var m;
    while ((m = re.exec(text)) !== null) {
      var trimmed = root.trimUtf16Span(text, last, m.index);
      if (trimmed.end > trimmed.start) spans.push(trimmed);
      last = m.index + m[0].length;
    }
    var tail = root.trimUtf16Span(text, last, text.length);
    if (tail.end > tail.start) spans.push(tail);
    return spans;
  }

  function chunkParagraphBased(text, params) {
    var maxParagraphs = Math.max(1, Number(params.max_paragraphs) || 3);
    var spans = paragraphSpans(text);
    var chunks = [];
    var n = 0;
    for (var i = 0; i < spans.length; i += maxParagraphs) {
      var group = spans.slice(i, i + maxParagraphs);
      chunks.push(
        root.chunkFromUtf16(
          "paragraph_based_" + n,
          text,
          group[0].start,
          group[group.length - 1].end,
          { chunker_used: "paragraph_based", source: params.source || "paste", paragraph_count: group.length }
        )
      );
      n += 1;
    }
    return chunks;
  }

  root.chunkParagraphBased = chunkParagraphBased;
})(typeof self !== "undefined" ? self : typeof global !== "undefined" ? global : window);
