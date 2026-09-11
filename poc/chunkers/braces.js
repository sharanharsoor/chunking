/* Brace / string / comment scanner used by in-tab code chunkers. */
(function (root) {
  function skipCommentOrString(text, i) {
    var c = text[i];
    var n = text[i + 1];
    if (c === "/" && n === "/") {
      i += 2;
      while (i < text.length && text[i] !== "\n") i += 1;
      return i;
    }
    if (c === "/" && n === "*") {
      i += 2;
      while (i < text.length - 1 && !(text[i] === "*" && text[i + 1] === "/")) i += 1;
      return Math.min(text.length, i + 2);
    }
    if (c === '"' || c === "'" || c === "`") {
      var q = c;
      i += 1;
      while (i < text.length) {
        if (text[i] === "\\") {
          i += 2;
          continue;
        }
        if (text[i] === q) return i + 1;
        i += 1;
      }
      return i;
    }
    return i;
  }

  function braceEnd(text, openAt) {
    var depth = 0;
    var i = openAt;
    while (i < text.length) {
      var next = skipCommentOrString(text, i);
      if (next !== i) {
        i = next;
        continue;
      }
      var c = text[i];
      if (c === "{") depth += 1;
      else if (c === "}") {
        depth -= 1;
        if (depth === 0) return i + 1;
      }
      i += 1;
    }
    return text.length;
  }

  function firstBrace(text, from) {
    var i = from;
    while (i < text.length) {
      var next = skipCommentOrString(text, i);
      if (next !== i) {
        i = next;
        continue;
      }
      if (text[i] === "{") return i;
      i += 1;
    }
    return -1;
  }

  function lineRecords(text) {
    var recs = [];
    var start = 0;
    for (var i = 0; i <= text.length; i++) {
      if (i === text.length || text[i] === "\n") {
        recs.push({ start: start, end: i, text: text.slice(start, i) });
        start = i + 1;
      }
    }
    return recs;
  }

  function spansByLineStart(text, testLine) {
    var lines = lineRecords(text);
    var spans = [];
    var skipUntil = -1;
    for (var i = 0; i < lines.length; i++) {
      if (lines[i].start < skipUntil) continue;
      if (!testLine(lines[i].text)) continue;
      var lookahead = lines[Math.min(lines.length - 1, i + 2)].end;
      var brace = firstBrace(text, lines[i].start);
      var end = brace >= 0 && brace <= lookahead ? braceEnd(text, brace) : lines[i].end;
      if (end <= lines[i].start) continue;
      spans.push({ start: lines[i].start, end: end });
      skipUntil = end;
    }
    return spans;
  }

  root.skipCommentOrString = skipCommentOrString;
  root.braceEnd = braceEnd;
  root.firstBrace = firstBrace;
  root.lineRecords = lineRecords;
  root.spansByLineStart = spansByLineStart;
})(typeof self !== "undefined" ? self : typeof global !== "undefined" ? global : window);
