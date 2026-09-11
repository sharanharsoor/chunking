/* python_code, javascript_code, css_code, go_code, java_code, c_cpp_code. Brace/indent cuts — not tree-sitter. */
(function (root) {
  function extra(name, params) {
    return { chunker_used: name, source: params.source || "paste", chunk_by: params.chunk_by || "function" };
  }

  function chunkPythonCode(text, params) {
    var lines = root.lineRecords(text);
    var defRe = /^\s*(async\s+)?def\s+\w+|^\s*class\s+\w+/;
    var decRe = /^\s*@\w/;
    var spans = [];
    function indent(s) {
      var m = s.match(/^[ \t]*/);
      return m ? m[0].length : 0;
    }
    function blank(s) {
      return !s.trim();
    }
    var i = 0;
    while (i < lines.length) {
      if (!defRe.test(lines[i].text)) {
        i += 1;
        continue;
      }
      var ind = indent(lines[i].text);
      var startLine = i;
      var j = i;
      while (j > 0 && decRe.test(lines[j - 1].text) && indent(lines[j - 1].text) === ind) j -= 1;
      startLine = j;
      var k = i + 1;
      while (k < lines.length) {
        if (blank(lines[k].text)) {
          k += 1;
          continue;
        }
        if (indent(lines[k].text) <= ind) break;
        k += 1;
      }
      spans.push({ start: lines[startLine].start, end: lines[k - 1] ? lines[k - 1].end : text.length });
      i = k;
    }
    return root.chunksFromSpans("python_code_", text, spans, extra("python_code", params));
  }

  function chunkJavascriptCode(text, params) {
    var re = /^\s*(export\s+)?(default\s+)?((async\s+)?function\s+\w+|(async\s+)?function\s*\(|class\s+\w+|(const|let|var)\s+\w+\s*=\s*(async\s+)?(function|\([^)]*\)\s*=>))/;
    return root.chunksFromSpans(
      "javascript_code_",
      text,
      root.spansByLineStart(text, function (line) { return re.test(line); }),
      extra("javascript_code", params)
    );
  }

  function chunkCssCode(text, params) {
    var spans = [];
    var i = 0;
    while (i < text.length) {
      var next = root.skipCommentOrString(text, i);
      if (next !== i) {
        i = next;
        continue;
      }
      if (/\s/.test(text[i])) {
        i += 1;
        continue;
      }
      var start = i;
      var brace = root.firstBrace(text, i);
      if (brace < 0) break;
      var end = root.braceEnd(text, brace);
      spans.push({ start: start, end: end });
      i = end;
    }
    return root.chunksFromSpans("css_code_", text, spans, extra("css_code", Object.assign({}, params, { chunk_by: "rule" })));
  }

  function chunkGoCode(text, params) {
    var re = /^\s*func\s+/;
    return root.chunksFromSpans(
      "go_code_",
      text,
      root.spansByLineStart(text, function (line) { return re.test(line); }),
      extra("go_code", params)
    );
  }

  function chunkJavaCode(text, params) {
    var classRe = /^\s*(public|private|protected)?\s*(abstract|final)?\s*class\s+\w+/;
    var methodRe = /^\s*(public|private|protected|static|final|synchronized|native|abstract)[\w\s<>,\[\]]+\([^;]*\)\s*\{?/;
    return root.chunksFromSpans(
      "java_code_",
      text,
      root.spansByLineStart(text, function (line) { return classRe.test(line) || methodRe.test(line); }),
      extra("java_code", params)
    );
  }

  function chunkCppCode(text, params) {
    var re = /^\s*[\w:\s\*&<>,]+\s+\w+\s*\([^;]*\)\s*(const)?\s*\{?/;
    return root.chunksFromSpans(
      "c_cpp_code_",
      text,
      root.spansByLineStart(text, function (line) {
        if (/^\s*(if|for|while|switch|else|catch)\b/.test(line)) return false;
        return re.test(line) && line.indexOf(";") === -1;
      }),
      extra("c_cpp_code", params)
    );
  }

  root.chunkPythonCode = chunkPythonCode;
  root.chunkJavascriptCode = chunkJavascriptCode;
  root.chunkCssCode = chunkCssCode;
  root.chunkGoCode = chunkGoCode;
  root.chunkJavaCode = chunkJavaCode;
  root.chunkCppCode = chunkCppCode;
})(typeof self !== "undefined" ? self : typeof global !== "undefined" ? global : window);
