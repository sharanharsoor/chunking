#!/usr/bin/env node
/* ponytail: smallest check that fails if a lab chunker breaks offsets or grouping. */
var fs = require("fs");
var path = require("path");
var vm = require("vm");

var root = { TextEncoder: TextEncoder, TextDecoder: TextDecoder, console: console };
root.self = root;
root.console = console;
var ctx = vm.createContext(root);
["offsets.js", "braces.js", "fixed_size.js", "sentence.js", "paragraph.js", "overlapping.js", "markdown.js", "csv.js", "json.js", "words.js", "xml.js", "code.js", "rolling.js", "fastcdc.js", "recursive_character.js", "token_based.js", "regex_custom.js", "tiktoken_bundle.js"].forEach(function (name) {
  var file = path.join(__dirname, name);
  vm.runInContext(fs.readFileSync(file, "utf8"), ctx, { filename: name });
});

var failed = 0;
function eq(name, got, want) {
  if (got !== want) {
    failed += 1;
    console.error("FAIL", name, "got", got, "want", want);
  }
}

var paras = ctx.chunkParagraphBased("one\n\ntwo\n\nthree", { max_paragraphs: 2 });
eq("paragraph count", paras.length, 2);
eq("paragraph first end", paras[0].end, ctx.utf16ToScalar("one\n\ntwo\n\nthree", "one\n\ntwo".length));

var md = ctx.chunkMarkdown("# A\nhello\n## B\nworld\n### Deep\nstill B", { header_level: 2 });
eq("markdown sections", md.length, 2);
eq("markdown first starts at title", md[0].content.indexOf("# A") === 0, true);

var csvText = "h1,h2\na,1\nb,2\nc,3\nd,4";
var csv = ctx.chunkCsv(csvText, { rows_per_chunk: 2, preserve_headers: true });
eq("csv chunks", csv.length, 2);
eq("csv first includes header", csv[0].content.indexOf("h1,h2") === 0, true);
eq("csv later content is data rows", csv[1].content.indexOf("c,3") === 0, true);
eq("csv later offsets skip header", csv[1].start > 0, true);
eq("csv header kept in metadata", csv[1].metadata.csv_header.indexOf("h1,h2") === 0, true);
eq("csv first row span", csv[0].metadata.csv_start_row === 1 && csv[0].metadata.csv_end_row === 2, true);
eq("csv later row span", csv[1].metadata.csv_start_row === 3 && csv[1].metadata.csv_end_row === 4, true);

var jsonText = '[{"a":1},{"a":2},{"a":3}]';
var json = ctx.chunkJson(jsonText, { objects_per_chunk: 2 });
eq("json chunks", json.length, 2);

var jsonl = ctx.chunkJson('{"a":1}\n{"a":2}\n{"a":3}\n', { objects_per_chunk: 1 });
eq("jsonl chunks", jsonl.length, 3);

var ov = ctx.chunkOverlappingWindow("abcdefghij", { window_size: 4, step_size: 2, window_unit: "characters" });
eq("overlap count", ov.length, 5);
eq("overlap first", ov[0].content, "abcd");
eq("overlap second", ov[1].content, "cdef");
eq("overlap last", ov[4].content, "ij");

var emoji = "hi 👩\u200d💻 z";
var fx = ctx.chunkFixedSize(emoji, { chunk_size: 4, overlap_size: 0 });
eq("scalar emoji not split wrong", fx[0].end, 4);

var fxo = ctx.chunkFixedSize("abcdefghij", { chunk_size: 4, overlap_size: 2 });
eq("fixed overlap count", fxo.length, 3);
eq("fixed overlap second", fxo[1].content, "cdefgh");

var words = ctx.chunkFixedLengthWord("one two three four five", { words_per_chunk: 2, overlap_words: 0 });
eq("word chunks", words.length, 3);

var py = ctx.chunkPythonCode("def a():\n    return 1\n\ndef b():\n    return 2\n", {});
eq("python defs", py.length, 2);

var js = ctx.chunkJavascriptCode("function a() { return 1; }\nfunction b() { return 2; }\n", {});
eq("js functions", js.length, 2);

var css = ctx.chunkCssCode("a { color: red; }\nb { color: blue; }\n", {});
eq("css rules", css.length, 2);

var html = ctx.chunkXmlHtml("<article>one</article><article>two</article>", { chunk_by: "semantic" });
eq("html articles", html.length, 2);

var rh = ctx.chunkRollingHash("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789", {
  min_chunk_size: 8, target_chunk_size: 16, max_chunk_size: 24, window_size: 8
});
eq("rolling splits", rh.length > 1, true);
eq("rolling covers", rh[0].start === 0 && rh[rh.length - 1].end > 50, true);

var cdcText = new Array(80).join("abcdefghij");
var cdc = ctx.chunkFastCdc(cdcText, { min_chunk_size: 32, avg_chunk_size: 64, max_chunk_size: 128, mask_bits: 6, hash_algorithm: "gear" });
eq("fastcdc splits", cdc.length > 1, true);
eq("fastcdc covers bytes", cdc[0].metadata.start_byte === 0 && cdc[cdc.length - 1].metadata.end_byte === new TextEncoder().encode(cdcText).length, true);
eq("fastcdc extras are distinct", cdc[0].metadata !== cdc[1].metadata, true);
eq("fastcdc first start_byte stays 0", cdc[0].metadata.start_byte, 0);

var rec = ctx.chunkRecursiveCharacter("Hello world.\n\nSecond paragraph lives here.\n\nThird.", { chunk_size: 40, overlap_size: 0 });
eq("recursive_character splits", rec.length > 1, true);
eq("recursive_character starts at 0", rec[0].start, 0);
eq("recursive_character first", rec[0].content, "Hello world.");
var recOvText = "Hello world.\n\nThis is a second paragraph that should stay together until the size cap.\n\nSupercalifragilisticexpialidociousAndMore";
var recOv = ctx.chunkRecursiveCharacter(recOvText, { chunk_size: 40, overlap_size: 20 });
var recOvHit = false;
for (var ri = 1; ri < recOv.length; ri++) {
  if (recOv[ri].start < recOv[ri - 1].end) recOvHit = true;
}
eq("recursive_character overlap", recOvHit, true);

var ranks = JSON.parse(fs.readFileSync(path.join(__dirname, "tiktoken", "cl100k_base.json"), "utf8"));
var enc = new ctx.JsTiktoken.Tiktoken(ranks);
var tok = ctx.chunkTokenBased("hello world hello world hello world hello world", { tokens_per_chunk: 4, overlap_tokens: 0, min_chunk_tokens: 1 }, enc);
eq("token_based splits", tok.length > 1, true);
eq("token_based starts at 0", tok[0].start, 0);
eq("token_based has token_count", tok[0].metadata.token_count > 0, true);
eq("token_based first tokens", tok[0].metadata.token_count, 4);
var tokOv = ctx.chunkTokenBased("hello world hello world hello world hello world", { tokens_per_chunk: 4, overlap_tokens: 2, min_chunk_tokens: 1 }, enc);
eq("token_based overlap", tokOv.length > 1 && tokOv[1].start < tokOv[0].end, true);

var rx = ctx.chunkRegexCustom("# A\nhello\n\n# B\nworld\n", { pattern: "^# ", multiline: true });
eq("regex_custom splits", rx.length, 2);
eq("regex_custom first", rx[0].content.indexOf("# A") === 0, true);
eq("regex_custom second", rx[1].content.indexOf("# B") === 0, true);

if (failed) {
  console.error(failed + " checks failed");
  process.exit(1);
}
console.log("ok");
