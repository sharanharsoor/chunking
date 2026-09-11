#!/usr/bin/env node
/* Compare in-tab JS chunkers to Python golden expected.json. */
var fs = require("fs");
var path = require("path");
var vm = require("vm");
var crypto = require("crypto");

var REPO = path.resolve(__dirname, "..");
var FIXTURES = path.join(REPO, "fixtures");
var CHUNKERS = path.join(REPO, "poc", "chunkers");

var RUNNERS = {
  fixed_size: "chunkFixedSize",
  sentence_based: "chunkSentenceBased",
  csv_chunker: "chunkCsv",
  paragraph_based: "chunkParagraphBased",
  overlapping_window: "chunkOverlappingWindow",
  markdown_chunker: "chunkMarkdown",
  json_chunker: "chunkJson",
  fixed_length_word: "chunkFixedLengthWord",
  fastcdc: "chunkFastCdc",
  recursive_character: "chunkRecursiveCharacter",
  token_based: "chunkTokenBased",
  regex_custom: "chunkRegexCustom",
};

/* Python rebuilds these; compare grouping metadata, not content bytes. */
var META_FIELDS = {
  csv_chunker: ["csv_start_row", "csv_end_row", "csv_row_count"],
  json_chunker: ["json_start_index", "json_end_index", "json_object_count"],
  paragraph_based: ["paragraph_count"],
  fixed_length_word: ["start_word_index", "end_word_index", "word_count"],
  token_based: ["token_count", "start_token_index"],
};

var SKIP_CONTENT = {
  csv_chunker: true,
  json_chunker: true,
  paragraph_based: true,
  fastcdc: true,
};

var ctx = { TextEncoder: TextEncoder, TextDecoder: TextDecoder, console: console };
ctx.self = ctx;
vm.createContext(ctx);
[
  "offsets.js",
  "braces.js",
  "token_packing.js",
  "fixed_size.js",
  "sentence.js",
  "paragraph.js",
  "overlapping.js",
  "markdown.js",
  "csv.js",
  "json.js",
  "words.js",
  "fastcdc.js",
  "recursive_character.js",
  "token_based.js",
  "regex_custom.js",
  "tiktoken_bundle.js",
].forEach(function (name) {
  vm.runInContext(fs.readFileSync(path.join(CHUNKERS, name), "utf8"), ctx, { filename: name });
});

var cl100k = JSON.parse(fs.readFileSync(path.join(CHUNKERS, "tiktoken", "cl100k_base.json"), "utf8"));
var tokenEnc = new ctx.JsTiktoken.Tiktoken(cl100k);

function sha256(buf) {
  return crypto.createHash("sha256").update(buf).digest("hex");
}

function iterFixtures(root) {
  var out = [];
  function walk(dir) {
    fs.readdirSync(dir, { withFileTypes: true }).forEach(function (ent) {
      var p = path.join(dir, ent.name);
      if (ent.isDirectory()) walk(p);
      else if (ent.name === "params.json") out.push(path.dirname(p));
    });
  }
  if (fs.existsSync(root)) walk(root);
  return out.sort();
}

function fail(msg) {
  console.error("FAIL " + msg);
  failed += 1;
}

var failed = 0;
var ran = 0;
iterFixtures(FIXTURES).forEach(function (folder) {
  var paramsDoc = JSON.parse(fs.readFileSync(path.join(folder, "params.json"), "utf8"));
  var strategy = paramsDoc.strategy;
  if (paramsDoc.python_only) {
    console.log("skip (Python-only golden) " + path.relative(REPO, folder));
    return;
  }
  var fnName = RUNNERS[strategy];
  if (!fnName) {
    console.log("skip (Python-only golden) " + path.relative(REPO, folder));
    return;
  }
  var inputPath = fs.existsSync(path.join(folder, "input.txt"))
    ? path.join(folder, "input.txt")
    : path.join(folder, "input.bin");
  var expectedPath = path.join(folder, "expected.json");
  if (!fs.existsSync(expectedPath)) {
    fail(path.relative(REPO, folder) + ": missing expected.json (python tools/run_fixture.py --write)");
    return;
  }
  var fileBuf = fs.readFileSync(inputPath);
  if (path.basename(inputPath) === "input.bin") {
    fail(path.relative(REPO, folder) + ": binary fixtures are Python-only");
    return;
  }
  var text = fileBuf.toString("utf8");
  var expected = JSON.parse(fs.readFileSync(expectedPath, "utf8"));
  var params = Object.assign({ source: "input.txt" }, paramsDoc.params || {});
  if (params.max_tokens != null || params.contextualize || params.window_unit === "tokens") {
    console.log("skip (Python-only golden) " + path.relative(REPO, folder));
    return;
  }
  var got = ctx[fnName](text, params, tokenEnc);
  if (!Array.isArray(got)) {
    fail(path.relative(REPO, folder) + ": " + fnName + " did not return an array");
    return;
  }
  ran += 1;
  var rel = path.relative(REPO, folder);
  var before = failed;
  var want = expected.chunks || [];
  if (got.length !== want.length) {
    fail(rel + ": chunk count js=" + got.length + " py=" + want.length);
    return;
  }
  var src = expected.source || {};
  if (src.sha256) {
    var actualHash = sha256(fs.readFileSync(inputPath));
    if (actualHash !== src.sha256) fail(rel + ": input.txt sha256 drifted from expected.json");
  }
  var metaKeys = META_FIELDS[strategy] || [];
  for (var i = 0; i < want.length; i++) {
    var e = want[i];
    var g = got[i];
    var prefix = rel + " chunk " + i;
    if (e.start != null && g.start !== e.start) fail(prefix + " start js=" + g.start + " py=" + e.start);
    if (e.end != null && g.end !== e.end) fail(prefix + " end js=" + g.end + " py=" + e.end);
    if (metaKeys.length) {
      var extra = (e.metadata && e.metadata.extra) || {};
      var gm = g.metadata || {};
      metaKeys.forEach(function (k) {
        if (extra[k] != null && gm[k] !== extra[k]) fail(prefix + " " + k + " js=" + gm[k] + " py=" + extra[k]);
      });
    }
    if (!SKIP_CONTENT[strategy] && e.content != null && g.content !== e.content) {
      fail(prefix + " content mismatch");
    }
    if (strategy === "fastcdc") {
      var extra = (e.metadata && e.metadata.extra) || {};
      var sb = g.metadata && g.metadata.start_byte;
      var eb = g.metadata && g.metadata.end_byte;
      if (sb == null || eb == null) fail(prefix + " missing start_byte/end_byte");
      else if (extra.sha256_hash && sha256(fileBuf.subarray(sb, eb)) !== extra.sha256_hash) {
        fail(prefix + " byte-range sha256 mismatch");
      }
    }
  }
  if (strategy === "fastcdc" && got.length) {
    if (got[0].metadata.start_byte !== 0) fail(rel + " fastcdc does not start at byte 0");
    if (got[got.length - 1].metadata.end_byte !== fileBuf.length) fail(rel + " fastcdc does not cover the file");
    for (var j = 1; j < got.length; j++) {
      if (got[j].metadata.start_byte !== got[j - 1].metadata.end_byte) fail(rel + " fastcdc gap at chunk " + j);
    }
  }
  if (failed === before) console.log("ok " + rel);
});

if (!ran) {
  console.error("no JS-covered fixtures found");
  process.exit(1);
}
if (failed) {
  console.error(failed + " checks failed");
  process.exit(1);
}
console.log("ok " + ran + " fixtures");
