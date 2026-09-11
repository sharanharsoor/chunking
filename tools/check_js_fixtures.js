#!/usr/bin/env node
/* Compare in-tab JS chunkers to Python golden expected.json (start/end/content). */
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
};

var ctx = { TextEncoder: TextEncoder, TextDecoder: TextDecoder, console: console };
ctx.self = ctx;
vm.createContext(ctx);
[
  "offsets.js",
  "braces.js",
  "fixed_size.js",
  "sentence.js",
  "csv.js",
].forEach(function (name) {
  vm.runInContext(fs.readFileSync(path.join(CHUNKERS, name), "utf8"), ctx, { filename: name });
});

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
  var fnName = RUNNERS[strategy];
  if (!fnName) return;
  var inputPath = fs.existsSync(path.join(folder, "input.txt"))
    ? path.join(folder, "input.txt")
    : path.join(folder, "input.bin");
  var expectedPath = path.join(folder, "expected.json");
  if (!fs.existsSync(expectedPath)) {
    fail(path.relative(REPO, folder) + ": missing expected.json (python tools/run_fixture.py --write)");
    return;
  }
  var text = fs.readFileSync(inputPath);
  if (path.basename(inputPath) === "input.bin") {
    fail(path.relative(REPO, folder) + ": binary fixtures are Python-only");
    return;
  }
  text = text.toString("utf8");
  var expected = JSON.parse(fs.readFileSync(expectedPath, "utf8"));
  var params = Object.assign({ source: "input.txt" }, paramsDoc.params || {});
  var got = ctx[fnName](text, params);
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
  for (var i = 0; i < want.length; i++) {
    var e = want[i];
    var g = got[i];
    var prefix = rel + " chunk " + i;
    if (e.start != null && g.start !== e.start) fail(prefix + " start js=" + g.start + " py=" + e.start);
    if (e.end != null && g.end !== e.end) fail(prefix + " end js=" + g.end + " py=" + e.end);
    if (strategy === "csv_chunker") {
      var extra = (e.metadata && e.metadata.extra) || {};
      var gm = g.metadata || {};
      ["csv_start_row", "csv_end_row", "csv_row_count"].forEach(function (k) {
        if (extra[k] != null && gm[k] !== extra[k]) fail(prefix + " " + k + " js=" + gm[k] + " py=" + extra[k]);
      });
    } else if (e.content != null && g.content !== e.content) {
      fail(prefix + " content mismatch");
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
