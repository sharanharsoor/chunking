#!/usr/bin/env node
/* ponytail: fails if vendored tree-sitter cannot parse two top-level defs. */
var path = require("path");
var TreeSitter = require("./tree-sitter.js");

var src = "def alpha():\n    return 1\n\ndef beta():\n    return 2\n";

TreeSitter.init({
  locateFile: function (name) {
    return path.join(__dirname, name);
  },
}).then(function () {
  return TreeSitter.Language.load(path.join(__dirname, "tree-sitter-python.wasm"));
}).then(function (lang) {
  var parser = new TreeSitter();
  parser.setLanguage(lang);
  var tree = parser.parse(src);
  var root = tree.rootNode;
  var names = [];
  for (var i = 0; i < root.namedChildCount; i++) {
    var child = root.namedChild(i);
    var inner = child.type === "function_definition" ? child : null;
    if (!inner) continue;
    var name = inner.childForFieldName("name");
    if (name) names.push(name.text);
  }
  if (names.join(",") !== "alpha,beta") {
    throw new Error("expected alpha,beta got " + names.join(","));
  }
  console.log("tree-sitter python ok");
}).catch(function (err) {
  console.error(String(err && err.stack ? err.stack : err));
  process.exit(1);
});
