/* semantic: consecutive-sentence cosine. Embeddings come from Transformers.js MiniLM in the worker. */
(function (root) {
  function cosine(a, b) {
    var dot = 0;
    var na = 0;
    var nb = 0;
    var i;
    var n = Math.min(a.length, b.length);
    for (i = 0; i < n; i++) {
      dot += a[i] * b[i];
      na += a[i] * a[i];
      nb += b[i] * b[i];
    }
    if (!na || !nb) return 0;
    var sim = dot / (Math.sqrt(na) * Math.sqrt(nb));
    if (sim < 0) return 0;
    if (sim > 1) return 1;
    return sim;
  }

  function chunkSemanticFromVectors(text, sentences, vectors, params) {
    params = params || {};
    var threshold = params.similarity_threshold == null ? 0.7 : Number(params.similarity_threshold);
    var minN = Math.max(1, Number(params.min_chunk_sentences) || 2);
    var maxN = Math.max(minN, Number(params.max_chunk_sentences) || 15);
    var source = params.source || "paste";
    if (!sentences.length) {
      return [root.chunkFromUtf16("semantic_0", text, 0, text.length, {
        chunker_used: "semantic",
        source: source,
        sentence_count: 0,
        embedding_model: params.embedding_model || "Xenova/all-MiniLM-L6-v2",
      })];
    }
    var spans = root.locateSentences(text, sentences);
    var similarities = [];
    var i;
    for (i = 0; i < vectors.length - 1; i++) similarities.push(cosine(vectors[i], vectors[i + 1]));
    var starts = [0];
    for (i = 0; i < similarities.length; i++) {
      if (similarities[i] < threshold && i + 1 - starts[starts.length - 1] >= minN) {
        starts.push(i + 1);
      }
    }
    if (starts[starts.length - 1] < sentences.length) starts.push(sentences.length);
    var ranges = [];
    for (i = 0; i < starts.length - 1; i++) {
      var a = starts[i];
      var b = starts[i + 1];
      while (b - a > maxN) {
        ranges.push([a, a + maxN]);
        a += maxN;
      }
      if (b > a) ranges.push([a, b]);
    }
    return ranges.map(function (pair, idx) {
      var lo = pair[0];
      var hi = pair[1] - 1;
      var utfStart = text.indexOf(sentences[lo]);
      var utfEnd = text.indexOf(sentences[hi], utfStart < 0 ? 0 : utfStart);
      if (utfStart < 0) utfStart = 0;
      if (utfEnd < 0) utfEnd = utfStart;
      utfEnd += sentences[hi].length;
      if (spans[lo] && spans[hi] && root.chunkFromScalar) {
        return root.chunkFromScalar("semantic_" + idx, text, spans[lo][0], spans[hi][1], {
          chunker_used: "semantic",
          source: source,
          sentence_count: pair[1] - pair[0],
          start_sentence_index: lo,
          end_sentence_index: hi,
          embedding_model: params.embedding_model || "Xenova/all-MiniLM-L6-v2",
          similarity_threshold: threshold,
        });
      }
      return root.chunkFromUtf16("semantic_" + idx, text, utfStart, utfEnd, {
        chunker_used: "semantic",
        source: source,
        sentence_count: pair[1] - pair[0],
        start_sentence_index: lo,
        end_sentence_index: hi,
        embedding_model: params.embedding_model || "Xenova/all-MiniLM-L6-v2",
        similarity_threshold: threshold,
      });
    });
  }

  function chunkSemantic(text, params, embed) {
    params = params || {};
    var sentences = root.splitSentencesSimpleV1(text);
    if (!sentences.length) {
      return Promise.resolve(chunkSemanticFromVectors(text, [], [], params));
    }
    if (Array.isArray(embed)) {
      return chunkSemanticFromVectors(text, sentences, embed, params);
    }
    return Promise.resolve(embed(sentences)).then(function (vectors) {
      return chunkSemanticFromVectors(text, sentences, vectors, params);
    });
  }

  root.cosine = cosine;
  root.chunkSemanticFromVectors = chunkSemanticFromVectors;
  root.chunkSemantic = chunkSemantic;
})(typeof self !== "undefined" ? self : typeof global !== "undefined" ? global : window);
