/* xml_html_chunker. Sibling blocks for semantic or tag names — no nested duplicates. */
(function (root) {
  var SEMANTIC = "header,nav,main,section,article,aside,footer,figure,details".split(",");
  var TAGS = "div,section,article,p,h1,h2,h3,h4,h5,h6,ul,ol,li,table,tr,pre,blockquote,item".split(",");

  function tagSet(list) {
    var o = {};
    for (var i = 0; i < list.length; i++) o[list[i]] = true;
    return o;
  }

  function escapeRe(s) {
    return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  }

  function closeEnd(text, name, from) {
    var re = new RegExp("<\\/?" + escapeRe(name) + "\\b[^>]*\\/?>", "gi");
    re.lastIndex = from;
    var depth = 1;
    var m;
    while ((m = re.exec(text)) !== null) {
      var selfClose = /\/\s*>$/.test(m[0]);
      var closing = m[0].charAt(1) === "/";
      if (closing) {
        depth -= 1;
        if (depth === 0) return m.index + m[0].length;
      } else if (!selfClose) {
        depth += 1;
      }
    }
    return text.length;
  }

  function tagSpans(text, names) {
    var want = tagSet(names);
    var re = /<\/?([A-Za-z][\w:.-]*)\b[^>]*\/?>/g;
    var spans = [];
    var skipUntil = -1;
    var m;
    while ((m = re.exec(text)) !== null) {
      if (m.index < skipUntil) continue;
      if (m[0].charAt(1) === "/") continue;
      var name = m[1].toLowerCase();
      if (!want[name]) continue;
      var end;
      if (/\/\s*>$/.test(m[0])) end = m.index + m[0].length;
      else end = closeEnd(text, name, m.index + m[0].length);
      spans.push({ start: m.index, end: end });
      skipUntil = end;
    }
    return spans;
  }

  function chunkXmlHtml(text, params) {
    var mode = params.chunk_by || "semantic";
    var names = mode === "tags" ? TAGS : SEMANTIC;
    var extra = { chunker_used: "xml_html_chunker", source: params.source || "paste", chunk_by: mode };
    var spans = tagSpans(text, names);
    if (!spans.length && mode === "semantic") spans = tagSpans(text, TAGS);
    return root.chunksFromSpans("xml_html_chunker_", text, spans, extra);
  }

  root.chunkXmlHtml = chunkXmlHtml;
})(typeof self !== "undefined" ? self : typeof global !== "undefined" ? global : window);
