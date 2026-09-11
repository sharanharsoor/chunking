/* Unicode-scalar fixed_size. Mirrors Python _chunk_by_characters (unit=character, no boundary preserve). */
(function (root) {
  function scalars(text) {
    return Array.from(text);
  }

  function chunkFixedSize(text, params) {
    const chunkSize = Math.max(1, Number(params.chunk_size) || 1024);
    const overlap = Math.max(0, Number(params.overlap_size) || 0);
    if (overlap >= chunkSize) {
      throw new Error("overlap_size must be less than chunk_size");
    }
    const chars = scalars(text);
    const n = chars.length;
    const chunks = [];
    if (n === 0) return chunks;

    let startPos = 0;
    let i = 0;
    while (startPos < n) {
      const chunkStart = Math.max(0, startPos - overlap);
      const chunkEnd = Math.min(n, startPos + chunkSize);
      const content = chars.slice(chunkStart, chunkEnd).join("");
      chunks.push({
        id: "fixed_size_" + i,
        content: content,
        start: chunkStart,
        end: chunkEnd,
        size: content.length,
        offset_unit: "char",
        metadata: { chunker_used: "fixed_size", source: params.source || "paste" },
      });
      i += 1;
      startPos = overlap === 0 ? chunkEnd : startPos + chunkSize;
    }
    return chunks;
  }

  root.chunkFixedSize = chunkFixedSize;
})(typeof self !== "undefined" ? self : window);
