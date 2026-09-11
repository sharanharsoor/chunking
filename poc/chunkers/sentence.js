/* simple_v1 sentence_based. Port of chunking_strategy.strategies.text.sentence_based */
(function (root) {
  var PUA_L = "\uE000";
  var PUA_R = "\uE001";
  var ABBREV =
    "dr|mr|mrs|ms|mx|prof|sr|jr|vs|etc|inc|ltd|dept|approx|est|vol|fig|eq|st|nd|rd|th|e\\.g|i\\.e|u\\.s|u\\.k|ph\\.d|a\\.m|p\\.m";
  var PROTECT = [
    { src: "(?<![A-Za-z0-9])https?:\\/\\/\\S+", flags: "g" },
    { src: "(?<![A-Za-z0-9])[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,}", flags: "g" },
    { src: "\\.{3,}|…", flags: "g" },
    { src: "(?<!\\d)\\d+\\.\\d+", flags: "g" },
    { src: "(?<![A-Za-z0-9])(?:" + ABBREV + ")\\.", flags: "gi" },
  ];

  function normalizeNewlines(text) {
    return text.replace(/\r\n/g, "\n").replace(/\r/g, "\n");
  }

  function utf16ToScalar(str, utf16Index) {
    return Array.from(str.slice(0, utf16Index)).length;
  }

  function splitSentencesSimpleV1(text) {
    text = normalizeNewlines(text);
    var held = [];
    var protectedText = text;
    for (var p = 0; p < PROTECT.length; p++) {
      var re = new RegExp(PROTECT[p].src, PROTECT[p].flags);
      protectedText = protectedText.replace(re, function (m) {
        var i = held.length;
        held.push(m);
        return PUA_L + String(i).padStart(4, "0") + PUA_R;
      });
    }
    function restore(s) {
      return s.replace(/\uE000(\d{4})\uE001/g, function (_, n) {
        return held[parseInt(n, 10)];
      });
    }
    // ponytail: same as Python finditer — optional quote stays on the sentence
    var boundary = /[.!?]["']?(?=\s+[A-Z0-9"'(])/g;
    var sentences = [];
    var start = 0;
    var match;
    while ((match = boundary.exec(protectedText)) !== null) {
      var end = match.index + match[0].length;
      var piece = protectedText.slice(start, end);
      if (piece.trim()) sentences.push(restore(piece.trim()));
      start = end;
      var ws = protectedText.slice(start).match(/^\s+/);
      if (ws) start += ws[0].length;
    }
    var tail = protectedText.slice(start);
    if (tail.trim()) sentences.push(restore(tail.trim()));
    return sentences;
  }

  function locateSentences(text, sentences) {
    var spans = [];
    var pos = 0;
    for (var i = 0; i < sentences.length; i++) {
      var s = sentences[i];
      var idx = text.indexOf(s, pos);
      if (idx < 0) idx = text.indexOf(s);
      if (idx < 0) {
        var fallback = pos;
        spans.push([utf16ToScalar(text, fallback), utf16ToScalar(text, fallback + s.length)]);
        pos = fallback + s.length;
      } else {
        spans.push([utf16ToScalar(text, idx), utf16ToScalar(text, idx + s.length)]);
        pos = idx + s.length;
      }
    }
    return spans;
  }

  function maybeDictionaryLines(text, sentences) {
    if (sentences.length && !(sentences.length === 1 && text.length > 1000)) {
      return sentences;
    }
    var lines = text.replace(/\s+$/, "").split("\n");
    if (lines.length > 5) {
      var short = 0;
      for (var i = 0; i < lines.length; i++) {
        if (lines[i].trim().split(/\s+/).filter(Boolean).length <= 2) short++;
      }
      if (short / lines.length > 0.7) {
        return lines.map(function (l) { return l.trim(); }).filter(Boolean);
      }
    }
    return sentences;
  }

  function chunkSentenceBased(text, params) {
    var maxSentences = Math.max(1, Number(params.max_sentences) || 5);
    var minSentences = Math.max(1, Number(params.min_sentences) || 1);
    var maxChunkSize = Math.max(1, Number(params.max_chunk_size) || 2000);
    var overlapSentences = Math.max(0, Number(params.overlap_sentences) || 0);
    text = normalizeNewlines(text);
    var sentences = maybeDictionaryLines(text, splitSentencesSimpleV1(text));
    var spans = locateSentences(text, sentences);
    var chunks = [];
    var i = 0;
    var n = 0;
    var chars = Array.from(text);
    while (i < sentences.length) {
      var group = [];
      var idxs = [];
      var size = 0;
      var added = 0;
      while (i < sentences.length && added < maxSentences && size < maxChunkSize) {
        var sentence = sentences[i];
        if (size + sentence.length > maxChunkSize && group.length) break;
        group.push(sentence);
        idxs.push(i);
        size += sentence.length;
        added += 1;
        i += 1;
      }
      while (group.length < minSentences && i < sentences.length) {
        group.push(sentences[i]);
        idxs.push(i);
        i += 1;
      }
      if (group.length) {
        var start = spans[idxs[0]][0];
        var end = spans[idxs[idxs.length - 1]][1];
        var content = chars.slice(start, end).join("");
        chunks.push({
          id: "sentence_based_" + n,
          content: content,
          start: start,
          end: end,
          size: content.length,
          offset_unit: "char",
          metadata: {
            chunker_used: "sentence_based",
            source: params.source || "paste",
            sentence_spec: "simple_v1",
          },
        });
        n += 1;
        if (overlapSentences > 0 && i < sentences.length) {
          var overlapStart = Math.max(0, group.length - overlapSentences);
          i -= group.length - overlapStart;
        }
      }
    }
    return chunks;
  }

  root.splitSentencesSimpleV1 = splitSentencesSimpleV1;
  root.chunkSentenceBased = chunkSentenceBased;
})(typeof self !== "undefined" ? self : window);
