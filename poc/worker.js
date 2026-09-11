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
  "chunkers/rolling.js"
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
};

function runJob(text, job) {
  var fn = RUNNERS[job.strategy];
  if (!fn) throw new Error("UNSUPPORTED");
  var params = Object.assign({ source: "upload" }, job.params || {});
  return fn(text, params);
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
      try {
        var results = {};
        var jobs = msg.jobs || [];
        for (var i = 0; i < jobs.length; i++) {
          var job = jobs[i];
          results[job.key] = runJob(text, job);
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
          code: err.message === "UNSUPPORTED" ? "UNSUPPORTED" : "INTERNAL",
          message: String(err && err.message ? err.message : err),
        });
      }
    }).catch(function (err) {
      self.postMessage({ id: id, type: "error", code: "INTERNAL", message: String(err) });
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
