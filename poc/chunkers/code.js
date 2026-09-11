/* python_code, javascript_code, css_code, go_code, java_code, c_cpp_code.
   python_code: tree-sitter module-level def/class when a parser is passed, else indent.
   Other languages: brace/indent cuts. */
(function (root) {
  function extra(name, params) {
    return { chunker_used: name, source: params.source || "paste", chunk_by: params.chunk_by || "function" };
  }

  function lineRange(text, utf16Start, utf16End) {
    var lines = root.lineRecords(text);
    var first = -1;
    var last = -1;
    var i;
    for (i = 0; i < lines.length; i++) {
      if (lines[i].end <= utf16Start) continue;
      if (lines[i].start >= utf16End) break;
      if (first < 0) first = i;
      last = i;
    }
    if (first < 0) return { start: utf16Start, end: utf16End, lineStart: 1, lineEnd: 1 };
    while (last > first && !String(lines[last].text).trim()) last -= 1;
    return {
      start: lines[first].start,
      end: lines[last].end,
      lineStart: first + 1,
      lineEnd: last + 1,
    };
  }

  function pythonName(block) {
    var lines = String(block || "").split("\n");
    var i;
    for (i = 0; i < lines.length; i++) {
      var m = lines[i].match(/^\s*(?:async\s+)?def\s+(\w+)/);
      if (m) return { name: m[1], kind: "function" };
      m = lines[i].match(/^\s*class\s+(\w+)/);
      if (m) return { name: m[1], kind: "class" };
    }
    return { name: "unnamed", kind: "other" };
  }

  function pythonChunk(text, utf16Start, utf16End, params, parserUsed, index) {
    var span = lineRange(text, utf16Start, utf16End);
    var meta = pythonName(text.slice(span.start, Math.min(text.length, span.start + 200)));
    return root.chunkFromUtf16("python_code_" + index, text, span.start, span.end, {
      chunker_used: "python_code",
      source: params.source || "paste",
      chunk_by: params.chunk_by || "function",
      symbol_name: meta.name,
      symbol_kind: meta.kind,
      line_start: span.lineStart,
      line_end: span.lineEnd,
      parser: parserUsed,
    });
  }

  function chunkPythonIndent(text, params) {
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
      var j = i;
      while (j > 0 && decRe.test(lines[j - 1].text) && indent(lines[j - 1].text) === ind) j -= 1;
      var k = i + 1;
      while (k < lines.length) {
        if (blank(lines[k].text)) {
          k += 1;
          continue;
        }
        if (indent(lines[k].text) <= ind) break;
        k += 1;
      }
      spans.push({ start: lines[j].start, end: lines[k - 1] ? lines[k - 1].end : text.length });
      i = k;
    }
    if (!spans.length) {
      if (!String(text).trim()) return [];
      return [pythonChunk(text, 0, text.length, params, "indent", 0)];
    }
    return spans.map(function (s, idx) {
      return pythonChunk(text, s.start, s.end, params, "indent", idx);
    });
  }

  function unwrapPythonDef(node) {
    if (!node) return null;
    if (node.type === "function_definition" || node.type === "class_definition") return node;
    if (node.type === "decorated_definition") {
      var n = node.namedChildCount || 0;
      var i;
      for (i = 0; i < n; i++) {
        var inner = unwrapPythonDef(node.namedChild(i));
        if (inner) return inner;
      }
    }
    return null;
  }

  function chunkPythonTreeSitter(text, params, parser) {
    var tree = parser.parse(text);
    var rootNode = tree.rootNode;
    var chunks = [];
    var i;
    var n = rootNode.namedChildCount || 0;
    for (i = 0; i < n; i++) {
      var child = rootNode.namedChild(i);
      var inner = unwrapPythonDef(child);
      if (!inner) continue;
      var nameNode = inner.childForFieldName && inner.childForFieldName("name");
      var kind = inner.type === "class_definition" ? "class" : "function";
      var span = lineRange(text, child.startIndex, child.endIndex);
      chunks.push(root.chunkFromUtf16("python_code_" + chunks.length, text, span.start, span.end, {
        chunker_used: "python_code",
        source: params.source || "paste",
        chunk_by: params.chunk_by || "function",
        symbol_name: nameNode && nameNode.text ? nameNode.text : "unnamed",
        symbol_kind: kind,
        line_start: span.lineStart,
        line_end: span.lineEnd,
        parser: "tree-sitter",
      }));
    }
    if (tree.delete) tree.delete();
    return chunks.length ? chunks : null;
  }

  function chunkPythonCode(text, params, _enc, parser) {
    if (parser && typeof parser.parse === "function") {
      try {
        var treeChunks = chunkPythonTreeSitter(text, params || {}, parser);
        if (treeChunks && treeChunks.length) return treeChunks;
      } catch (err) {
        /* indent fallback */
      }
    }
    return chunkPythonIndent(text, params || {});
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
