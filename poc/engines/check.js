var assert = require("assert");
var extract = require("./extract.js");

assert.equal(extract.sniffKind("a.bin", new Uint8Array([0x25, 0x50, 0x44, 0x46, 0x2d])), "pdf");
assert.equal(extract.sniffKind("a.pdf", new Uint8Array([0x25, 0x50, 0x44, 0x46])), "pdf");
assert.equal(extract.sniffKind("n.docx", new Uint8Array([0x50, 0x4B, 0x03, 0x04, 0x77, 0x6F, 0x72, 0x64, 0x2F, 0x64, 0x6F, 0x63, 0x75, 0x6D, 0x65, 0x6E, 0x74])), "docx");
assert.equal(extract.sniffKind("old.doc", new Uint8Array([0xD0, 0xCF, 0x11, 0xE0, 0xA1, 0xB1, 0x1A, 0xE1])), "ole");
assert.equal(extract.sniffKind("pic.png", new Uint8Array([0x89, 0x50, 0x4E, 0x47])), "media");

var line = extract.itemsToText([
  { str: "Hello", transform: [1, 0, 0, 1, 10, 700], width: 28 },
  { str: "world", transform: [1, 0, 0, 1, 42, 700], width: 30 },
  { str: "Next", transform: [1, 0, 0, 1, 10, 680], width: 22 },
]);
assert.equal(line, "Hello world\nNext");

var cols = [];
for (var i = 0; i < 8; i++) {
  cols.push({ str: "L" + i, transform: [1, 0, 0, 1, 20, 500 - i * 10], width: 10 });
  cols.push({ str: "R" + i, transform: [1, 0, 0, 1, 320, 500 - i * 10], width: 10 });
}
assert.equal(extract.pageHasColumns(cols, 400), true);
assert.equal(extract.pageHasColumns(cols.slice(0, 4), 400), false);

console.log("extract sniff/layout ok");
