/* hierarchical recursive. Parent/child levels — not recursive_character. */
(function (root) {
  function defaultLevels() {
    return [
      {
        name: "paragraph",
        strategy: "paragraph",
        parameters: { min_paragraph_length: 50 },
        min_chunk_size: 50,
        max_chunk_size: 2500,
        target_chunk_size: 1200,
        quality_threshold: 0.7,
      },
      {
        name: "sentence",
        strategy: "sentence",
        parameters: { max_sentences: 4 },
        min_chunk_size: 50,
        max_chunk_size: 800,
        target_chunk_size: 400,
        quality_threshold: 0.7,
      },
      {
        name: "fixed",
        strategy: "fixed_size",
        parameters: { chunk_size: 200 },
        min_chunk_size: 50,
        max_chunk_size: 300,
        target_chunk_size: 200,
        quality_threshold: 0.7,
      },
    ];
  }

  function runnerFor(strategy) {
    if (strategy === "paragraph") return root.chunkParagraphBased;
    if (strategy === "sentence") return root.chunkSentenceBased;
    if (strategy === "fixed_size") return root.chunkFixedSize;
    if (strategy === "overlapping") return root.chunkOverlappingWindow;
    return root.chunkSentenceBased;
  }

  function qualityOf(chunk, level) {
    var content = chunk.content || "";
    var target = Number(level.target_chunk_size) || 500;
    var maxSize = Number(level.max_chunk_size) || 2000;
    var sizeScore = Math.min(1, content.length / target);
    if (content.length > maxSize) sizeScore *= 0.5;
    var coherence = Math.min(1, content.split(". ").length / 5);
    var boundary = 1;
    var trimmed = content.replace(/^\s+|\s+$/g, "");
    if (trimmed && !/[.!?\n]$/.test(trimmed)) boundary = 0.8;
    return sizeScore * 0.4 + coherence * 0.4 + boundary * 0.2;
  }

  function walk(content, level, parentId, hierarchyPath, ctx) {
    if (level >= ctx.levels.length || level >= ctx.maxDepth) return [];
    if (!String(content).replace(/^\s+|\s+$/g, "")) return [];
    var current = ctx.levels[level];
    var minSize = current.min_chunk_size != null ? Number(current.min_chunk_size) : 50;
    if (ctx.adaptive && content.length < minSize) {
      if (level + 1 < ctx.levels.length) return walk(content, level + 1, parentId, hierarchyPath, ctx);
      return [];
    }
    var strategy = current.strategy;
    var run = runnerFor(strategy);
    var innerParams = Object.assign({ source: ctx.source }, current.parameters || {});
    var inner;
    try {
      inner = run(content, innerParams, ctx.enc) || [];
    } catch (err) {
      if (level + 1 < ctx.levels.length) return walk(content, level + 1, parentId, hierarchyPath, ctx);
      return [];
    }
    var qThresh = current.quality_threshold != null ? Number(current.quality_threshold) : ctx.qualityThreshold;
    var out = [];
    for (var chunkIdx = 0; chunkIdx < inner.length; chunkIdx++) {
      var chunk = inner[chunkIdx];
      var currentPath = hierarchyPath ? hierarchyPath + "." + chunkIdx : String(chunkIdx);
      if (qualityOf(chunk, current) < qThresh) continue;
      var rec = {
        id: "recursive_" + currentPath.replace(/\./g, "_"),
        content: chunk.content,
        start: chunk.start,
        end: chunk.end,
        size: (chunk.content || "").length,
        offset_unit: "char",
        parent_id: parentId || null,
        children_ids: [],
        metadata: {
          chunker_used: "recursive",
          source: ctx.source,
          offset_unit: "char",
          level: level,
          hierarchy_path: currentPath,
          level_strategy: strategy,
          level_name: current.name || strategy,
        },
      };
      if (level + 1 < ctx.levels.length && level + 1 < ctx.maxDepth) {
        var kids = walk(chunk.content, level + 1, rec.id, currentPath, ctx);
        if (kids.length) {
          rec.children_ids = kids.map(function (k) { return k.id; });
          out = out.concat(kids);
        }
      }
      out.push(rec);
    }
    return out;
  }

  function chunkRecursive(text, params, enc) {
    params = params || {};
    var maxDepth = Math.max(1, Number(params.max_depth) || 3);
    var adaptive = params.adaptive_depth == null ? true : !!params.adaptive_depth;
    var qualityThreshold = params.quality_threshold == null ? 0.7 : Number(params.quality_threshold);
    var levels = (params.hierarchy_levels && params.hierarchy_levels.length)
      ? params.hierarchy_levels.slice()
      : defaultLevels();
    levels = levels.slice(0, maxDepth);
    if (!String(text || "").replace(/^\s+|\s+$/g, "")) return [];
    var got = walk(text, 0, null, "", {
      levels: levels,
      maxDepth: maxDepth,
      adaptive: adaptive,
      qualityThreshold: qualityThreshold,
      source: params.source || "paste",
      enc: enc,
    });
    if (!got.length) {
      return [{
        id: "recursive_fallback_0",
        content: text,
        start: 0,
        end: Array.from(text).length,
        size: text.length,
        offset_unit: "char",
        parent_id: null,
        children_ids: [],
        metadata: {
          chunker_used: "recursive",
          source: params.source || "paste",
          offset_unit: "char",
          level: 0,
          fallback_mode: true,
        },
      }];
    }
    return got;
  }

  function liftRecursiveOffsets(doc, chunks) {
    if (!chunks || !chunks.length) return chunks;
    var byId = {};
    for (var i = 0; i < chunks.length; i++) byId[chunks[i].id] = chunks[i];
    function abs(c) {
      if (c._docStart != null) return;
      var p = c.parent_id ? byId[c.parent_id] : null;
      if (!p) {
        if (c.start != null && c.end != null) {
          c._docStart = c.start;
          c._docEnd = c.end;
          return;
        }
        var idx = doc.indexOf(c.content || "");
        if (idx < 0) {
          c._docStart = 0;
          c._docEnd = Array.from(c.content || "").length;
        } else {
          c._docStart = Array.from(doc.slice(0, idx)).length;
          c._docEnd = Array.from(doc.slice(0, idx + (c.content || "").length)).length;
        }
        return;
      }
      abs(p);
      var base = p._docStart || 0;
      var relStart = c.start == null ? 0 : c.start;
      var relEnd = c.end == null ? relStart + Array.from(c.content || "").length : c.end;
      c._docStart = base + relStart;
      c._docEnd = base + relEnd;
    }
    for (var j = 0; j < chunks.length; j++) abs(chunks[j]);
    for (var k = 0; k < chunks.length; k++) {
      var c = chunks[k];
      if (c._docStart != null) {
        c.start = c._docStart;
        c.end = c._docEnd;
        delete c._docStart;
        delete c._docEnd;
      }
    }
    return chunks;
  }

  root.chunkRecursive = chunkRecursive;
  root.liftRecursiveOffsets = liftRecursiveOffsets;
})(typeof self !== "undefined" ? self : typeof global !== "undefined" ? global : window);
