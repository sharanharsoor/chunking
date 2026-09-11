/* recursive_character. Separator cascade — not hierarchical recursive. */
(function (root) {
  var DEFAULT_SEPARATORS = ["\n\n", "\n", " ", ""];

  function lengthOf(text) {
    return Array.from(text).length;
  }

  function splitKeep(text, separator) {
    if (separator === "") return Array.from(text);
    var parts = text.split(separator);
    var out = [];
    for (var i = 0; i < parts.length; i++) {
      if (parts[i] !== "") out.push(parts[i]);
    }
    return out;
  }

  function joinParts(parts, separator) {
    if (!parts.length) return null;
    var text = parts.join(separator);
    return text ? text : null;
  }

  function mergeSplits(splits, separator, chunkSize, overlapSize) {
    var sepLen = separator ? lengthOf(separator) : 0;
    var docs = [];
    var current = [];
    var total = 0;
    for (var i = 0; i < splits.length; i++) {
      var piece = splits[i];
      var plen = lengthOf(piece);
      if (total + plen + (current.length ? sepLen : 0) > chunkSize) {
        if (current.length) {
          var joined = joinParts(current, separator);
          if (joined != null) docs.push(joined);
          while (
            current.length &&
            (total > overlapSize ||
              (total + plen + (current.length ? sepLen : 0) > chunkSize && total > 0))
          ) {
            total -= lengthOf(current[0]) + (current.length > 1 ? sepLen : 0);
            current = current.slice(1);
          }
        }
      }
      current.push(piece);
      total += plen + (current.length > 1 ? sepLen : 0);
    }
    var last = joinParts(current, separator);
    if (last != null) docs.push(last);
    return docs;
  }

  function splitWith(piece, seps, chunkSize, overlapSize) {
    var separator = seps[seps.length - 1];
    var rest = [];
    var i;
    for (i = 0; i < seps.length; i++) {
      var sep = seps[i];
      if (sep === "") {
        separator = sep;
        rest = [];
        break;
      }
      if (piece.indexOf(sep) !== -1) {
        separator = sep;
        rest = seps.slice(i + 1);
        break;
      }
    }
    var splits = splitKeep(piece, separator);
    var out = [];
    var good = [];
    for (i = 0; i < splits.length; i++) {
      var part = splits[i];
      if (lengthOf(part) < chunkSize) {
        good.push(part);
        continue;
      }
      if (good.length) {
        out = out.concat(mergeSplits(good, separator, chunkSize, overlapSize));
        good = [];
      }
      if (!rest.length) out.push(part);
      else out = out.concat(splitWith(part, rest, chunkSize, overlapSize));
    }
    if (good.length) out = out.concat(mergeSplits(good, separator, chunkSize, overlapSize));
    return out;
  }

  function scalarIndexOf(text, piece, hint) {
    var chars = Array.from(text);
    var needle = Array.from(piece);
    if (!needle.length) return Math.max(0, hint);
    var n = chars.length;
    var m = needle.length;
    function scan(from) {
      var i, j, ok;
      for (i = Math.max(0, from); i <= n - m; i++) {
        ok = true;
        for (j = 0; j < m; j++) {
          if (chars[i + j] !== needle[j]) {
            ok = false;
            break;
          }
        }
        if (ok) return i;
      }
      return -1;
    }
    var idx = scan(hint);
    if (idx >= 0) return idx;
    if (hint > 0) return scan(0);
    return -1;
  }

  function chunkRecursiveCharacter(text, params) {
    var chunkSize = Math.max(1, Number(params.chunk_size) || 1000);
    var overlap = Number(params.overlap_size) || 0;
    if (overlap < 0) overlap = 0;
    if (overlap >= chunkSize) overlap = Math.max(0, chunkSize - 1);
    var seps = params.separators;
    if (!seps || !seps.length) seps = DEFAULT_SEPARATORS.slice();
    else seps = seps.slice();
    if (seps[seps.length - 1] !== "") seps.push("");
    if (!text) return [];
    var pieces = splitWith(text, seps, chunkSize, overlap);
    var hint = 0;
    var chunks = [];
    for (var i = 0; i < pieces.length; i++) {
      var piece = pieces[i];
      var start = scalarIndexOf(text, piece, hint);
      if (start < 0) start = hint;
      var end = start + lengthOf(piece);
      chunks.push({
        id: "recursive_character_" + i,
        content: piece,
        start: start,
        end: end,
        size: piece.length,
        offset_unit: "char",
        metadata: {
          chunker_used: "recursive_character",
          source: params.source || "paste",
          chunk_size: chunkSize,
          overlap_size: overlap,
        },
      });
      hint = overlap ? Math.max(start + 1, end - overlap) : end;
    }
    return chunks;
  }

  root.chunkRecursiveCharacter = chunkRecursiveCharacter;
  root.splitRecursiveCharacter = function (text, params) {
    return chunkRecursiveCharacter(text, params).map(function (c) { return c.content; });
  };
})(typeof self !== "undefined" ? self : typeof global !== "undefined" ? global : window);
