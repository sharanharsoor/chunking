importScripts(
  "chunkers/offsets.js",
  "chunkers/braces.js",
  "chunkers/fixed_size.js",
  "chunkers/sentence.js",
  "chunkers/paragraph.js",
  "chunkers/overlapping.js",
  "chunkers/markdown.js",
  "chunkers/csv.js",
  "chunkers/json.js",
  "chunkers/words.js",
  "chunkers/xml.js",
  "chunkers/code.js",
  "chunkers/rolling.js",
  "chunkers/fastcdc.js",
  "chunkers/recursive_character.js",
  "chunkers/token_based.js",
  "chunkers/regex_custom.js"
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
  token_based: chunkTokenBased,
  regex_custom: chunkRegexCustom,
};

var encPromise = null;

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

function annotateTokenCounts(chunks, enc) {
  if (!enc || !chunks) return;
  for (var i = 0; i < chunks.length; i++) {
    var m = chunks[i].metadata || (chunks[i].metadata = {});
    if (m.token_count != null) continue;
    m.token_count = enc.encode(chunks[i].content || "").length;
  }
}

function runJob(text, job, enc) {
  var fn = RUNNERS[job.strategy];
  if (!fn) throw new Error("UNSUPPORTED");
  var params = Object.assign({ source: "upload" }, job.params || {});
  var chunks = job.strategy === "token_based" ? fn(text, params, enc) : fn(text, params);
  if (enc) annotateTokenCounts(chunks, enc);
  return chunks;
}

self.onmessage = function (ev) {
  var msg = ev.data || {};
  var id = msg.id;
  try {
    var file = msg.file;
    if (!file) throw new Error("INTERNAL");
    if (file.size > 150 * 1024 * 1024) {
      self.postMessage({ id: id, type: "error", code: "FILE_TOO_LARGE", message: "The in-browser lab stops at 150 MB." });
      return;
    }
    file.text().then(function (text) {
      var jobs = msg.jobs || [];
      var needEnc = false;
      for (var i = 0; i < jobs.length; i++) {
        if (jobs[i].strategy === "token_based") needEnc = true;
      }
      // First paint never fetches ranks. After token_based once, self._enc stays and every later job gets token_count.
      var ready = needEnc || self._enc ? getEncoder() : Promise.resolve(self._enc || null);
      if (needEnc && !self._enc) {
        self.postMessage({ id: id, type: "progress", phase: "tokenizer" });
      }
      return ready.then(function (enc) {
        try {
          if (needEnc && !enc) throw new Error("TOKENIZER");
          var results = {};
          for (var i = 0; i < jobs.length; i++) {
            var job = jobs[i];
            results[job.key] = runJob(text, job, enc);
            self.postMessage({
              id: id,
              type: "progress",
              job: job.key,
              chunks: results[job.key].length,
            });
          }
          self.postMessage({ id: id, type: "result", text: text, results: results });
        } catch (err) {
          self.postMessage({
            id: id,
            type: "error",
            code: err.message === "UNSUPPORTED" ? "UNSUPPORTED" : err.message === "TOKENIZER" ? "TOKENIZER" : err.message === "INVALID_REGEX" ? "INVALID_REGEX" : "INTERNAL",
            message: err.message === "TOKENIZER"
              ? "Could not load cl100k_base. Serve this lab over http, not file://."
              : err.message === "INVALID_REGEX"
              ? "That regex is invalid in JavaScript."
              : String(err && err.message ? err.message : err),
          });
        }
      });
    }).catch(function (err) {
      self.postMessage({
        id: id,
        type: "error",
        code: err && err.message === "TOKENIZER" ? "TOKENIZER" : "INTERNAL",
        message: err && err.message === "TOKENIZER"
          ? "Could not load cl100k_base. Serve this lab over http, not file://."
          : String(err),
      });
    });
  } catch (err) {
    self.postMessage({
      id: id,
      type: "error",
      code: "INTERNAL",
      message: String(err && err.message ? err.message : err),
    });
  }
};
