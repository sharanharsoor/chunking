var ASSET = "gate6";
importScripts(
  "chunkers/offsets.js?" + ASSET,
  "chunkers/braces.js?" + ASSET,
  "chunkers/token_packing.js?" + ASSET,
  "chunkers/fixed_size.js?" + ASSET,
  "chunkers/sentence.js?" + ASSET,
  "chunkers/paragraph.js?" + ASSET,
  "chunkers/overlapping.js?" + ASSET,
  "chunkers/markdown.js?" + ASSET,
  "chunkers/csv.js?" + ASSET,
  "chunkers/json.js?" + ASSET,
  "chunkers/words.js?" + ASSET,
  "chunkers/xml.js?" + ASSET,
  "chunkers/code.js?" + ASSET,
  "chunkers/rolling.js?" + ASSET,
  "chunkers/fastcdc.js?" + ASSET,
  "chunkers/recursive_character.js?" + ASSET,
  "chunkers/recursive.js?" + ASSET,
  "chunkers/token_based.js?" + ASSET,
  "chunkers/regex_custom.js?" + ASSET,
  "chunkers/semantic.js?" + ASSET
);

var RUNNERS = {
  fixed_size: chunkFixedSize,
  sentence_based: chunkSentenceBased,
  paragraph_based: chunkParagraphBased,
  overlapping_window: chunkOverlappingWindow,
  fixed_length_word: chunkFixedLengthWord,
  markdown_chunker: chunkMarkdown,
  xml_html_chunker: chunkXmlHtml,
  csv_chunker: chunkCsv,
  json_chunker: chunkJson,
  python_code: chunkPythonCode,
  javascript_code: chunkJavascriptCode,
  css_code: chunkCssCode,
  go_code: chunkGoCode,
  java_code: chunkJavaCode,
  c_cpp_code: chunkCppCode,
  rolling_hash: chunkRollingHash,
  fastcdc: chunkFastCdc,
  recursive_character: chunkRecursiveCharacter,
  recursive: chunkRecursive,
  token_based: chunkTokenBased,
  regex_custom: chunkRegexCustom,
  semantic: chunkSemantic,
};

var encPromise = null;
var pyParserPromise = null;
var semanticPromise = null;

function getEncoder() {
  if (self._enc) return Promise.resolve(self._enc);
  if (!encPromise) {
    encPromise = fetch("chunkers/tiktoken/cl100k_base.json")
      .then(function (r) {
        if (!r.ok) throw new Error("TOKENIZER");
        return r.json();
      })
      .then(function (ranks) {
        if (!self.JsTiktoken) importScripts("chunkers/tiktoken_bundle.js");
        self._enc = new JsTiktoken.Tiktoken(ranks);
        return self._enc;
      })
      .catch(function (err) {
        encPromise = null;
        throw err && err.message === "TOKENIZER" ? err : new Error("TOKENIZER");
      });
  }
  return encPromise;
}

function getPythonParser(id) {
  if (self._pyParser) return Promise.resolve(self._pyParser);
  if (self._pyParser === false) return Promise.resolve(null);
  if (!pyParserPromise) {
    if (id != null) self.postMessage({ id: id, type: "progress", phase: "treesitter" });
    pyParserPromise = Promise.resolve()
      .then(function () {
        if (!self.TreeSitter) importScripts("engines/tree-sitter/tree-sitter.js?" + ASSET);
        return TreeSitter.init({
          locateFile: function (name) {
            if (/\.wasm$/.test(name)) return "engines/tree-sitter/tree-sitter.wasm";
            return name;
          },
        });
      })
      .then(function () {
        return TreeSitter.Language.load("engines/tree-sitter/tree-sitter-python.wasm");
      })
      .then(function (lang) {
        var parser = new TreeSitter();
        parser.setLanguage(lang);
        self._pyParser = parser;
        return parser;
      })
      .catch(function () {
        self._pyParser = false;
        pyParserPromise = null;
        return null;
      });
  }
  return pyParserPromise;
}

function getSemantic(id) {
  if (self._embed) return Promise.resolve(self._embed);
  if (!semanticPromise) {
    if (id != null) self.postMessage({ id: id, type: "progress", phase: "semantic" });
    semanticPromise = new Promise(function (resolve, reject) {
      var w;
      try {
        w = new Worker("engines/semantic/embed.js", { type: "module" });
      } catch (err) {
        semanticPromise = null;
        reject(new Error("SEMANTIC"));
        return;
      }
      var reqs = {};
      var next = 1;
      w.onmessage = function (ev) {
        var msg = ev.data || {};
        var waiter = reqs[msg.id];
        if (!waiter) return;
        delete reqs[msg.id];
        if (msg.type === "error") waiter.reject(new Error("SEMANTIC"));
        else waiter.resolve(msg);
      };
      w.onerror = function () {
        semanticPromise = null;
        reject(new Error("SEMANTIC"));
      };
      function call(type, extra) {
        var rid = String(next++);
        return new Promise(function (res, rej) {
          reqs[rid] = { resolve: res, reject: rej };
          w.postMessage(Object.assign({ id: rid, type: type }, extra || {}));
        });
      }
      call("warmup").then(function () {
        self._embed = function (texts) {
          return call("embed", { texts: texts }).then(function (msg) { return msg.vectors; });
        };
        resolve(self._embed);
      }).catch(function (err) {
        semanticPromise = null;
        reject(err && err.message === "SEMANTIC" ? err : new Error("SEMANTIC"));
      });
    });
  }
  return semanticPromise;
}

function annotateTokenCounts(chunks, enc) {
  if (!enc || !chunks) return;
  for (var i = 0; i < chunks.length; i++) {
    var m = chunks[i].metadata || (chunks[i].metadata = {});
    if (m.token_count != null) continue;
    m.token_count = enc.encode(chunks[i].content || "").length;
  }
}

function jobNeedsEncoder(job) {
  if (job.strategy === "token_based") return true;
  var p = job.params || {};
  if (Number(p.max_tokens) > 0) return true;
  if (p.window_unit === "tokens") return true;
  return false;
}

function runJob(text, job, enc, parser, embed) {
  var fn = RUNNERS[job.strategy];
  if (!fn) throw new Error("UNSUPPORTED");
  var params = Object.assign({ source: "upload" }, job.params || {});
  var chunks;
  if (job.strategy === "semantic") {
    chunks = fn(text, params, embed);
  } else if (job.strategy === "python_code") {
    chunks = fn(text, params, enc, parser);
  } else {
    chunks = fn(text, params, enc);
  }
  return Promise.resolve(chunks).then(function (out) {
    if (job.strategy === "recursive") liftRecursiveOffsets(text, out);
    if (enc) annotateTokenCounts(out, enc);
    return out;
  });
}

function runJobs(text, jobs, enc, parser, embed, id) {
  var results = {};
  var i = 0;
  function next() {
    if (i >= jobs.length) return Promise.resolve(results);
    var job = jobs[i++];
    return runJob(text, job, enc, parser, embed).then(function (chunks) {
      results[job.key] = chunks;
      self.postMessage({
        id: id,
        type: "progress",
        job: job.key,
        chunks: chunks.length,
      });
      return next();
    });
  }
  return next();
}

function fail(id, err) {
  var code = "INTERNAL";
  var message = String(err && err.message ? err.message : err);
  if (message === "UNSUPPORTED") code = "UNSUPPORTED";
  else if (message === "TOKENIZER") {
    code = "TOKENIZER";
    message = "Could not load cl100k_base. Serve this lab over http, not file://.";
  } else if (message === "INVALID_REGEX") {
    code = "INVALID_REGEX";
    message = "That regex is invalid in JavaScript.";
  } else if (message === "SEMANTIC") {
    code = "SEMANTIC";
    message = "Could not load MiniLM from Hugging Face. Serve this lab over http, not file://.";
  }
  self.postMessage({ id: id, type: "error", code: code, message: message });
}

self.onmessage = function (ev) {
  var msg = ev.data || {};
  var id = msg.id;
  try {
    if (msg.type === "load_semantic") {
      getSemantic(id).then(function () {
        self.postMessage({ id: id, type: "semantic_ready" });
      }).catch(function (err) { fail(id, err); });
      return;
    }
    var file = msg.file;
    if (file && file.size > 150 * 1024 * 1024) {
      self.postMessage({ id: id, type: "error", code: "FILE_TOO_LARGE", message: "The in-browser lab stops at 150 MB." });
      return;
    }
    var incoming = typeof msg.text === "string" ? Promise.resolve(msg.text) : null;
    if (!incoming) {
      if (!file) throw new Error("INTERNAL");
      incoming = file.text();
    }
    incoming.then(function (text) {
      var jobs = msg.jobs || [];
      var needEnc = false;
      var needPy = false;
      var needSem = false;
      for (var i = 0; i < jobs.length; i++) {
        if (jobNeedsEncoder(jobs[i])) needEnc = true;
        if (jobs[i].strategy === "python_code") needPy = true;
        if (jobs[i].strategy === "semantic") needSem = true;
      }
      var ready = Promise.resolve();
      if (needEnc || self._enc) ready = getEncoder();
      else ready = Promise.resolve(self._enc || null);
      if (needEnc && !self._enc) {
        self.postMessage({ id: id, type: "progress", phase: "tokenizer" });
      }
      return ready.then(function (enc) {
        if (needEnc && !enc) throw new Error("TOKENIZER");
        var parserP = needPy ? getPythonParser(id) : Promise.resolve(null);
        var embedP = needSem ? getSemantic(id) : Promise.resolve(null);
        return Promise.all([parserP, embedP]).then(function (pair) {
          return runJobs(text, jobs, enc, pair[0], pair[1], id);
        }).then(function (results) {
          self.postMessage({ id: id, type: "result", text: text, results: results });
        });
      });
    }).catch(function (err) {
      fail(id, err);
    });
  } catch (err) {
    fail(id, err);
  }
};
