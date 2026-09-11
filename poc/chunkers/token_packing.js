/* Greedy token packing. Port of chunking_strategy.core.token_packing. Never chars/4. */
(function (root) {
  function tokenWindows(nTokens, maxTokens, overlapTokens) {
    if (maxTokens < 1) throw new Error("max_tokens must be at least 1");
    if (overlapTokens < 0) throw new Error("overlap_tokens must be non-negative");
    if (overlapTokens >= maxTokens) throw new Error("overlap_tokens must be less than max_tokens");
    var step = Math.max(1, maxTokens - overlapTokens);
    var out = [];
    var start = 0;
    while (start < nTokens) {
      var end = Math.min(start + maxTokens, nTokens);
      out.push([start, end]);
      if (end >= nTokens) break;
      start += step;
    }
    return out;
  }

  function packUnitRanges(sizes, maxTokens, overlapTokens, maxUnits) {
    if (maxTokens < 1) throw new Error("max_tokens must be at least 1");
    if (overlapTokens < 0) overlapTokens = 0;
    var n = sizes.length;
    var ranges = [];
    var i = 0;
    while (i < n) {
      var start = i;
      var used = 0;
      var units = 0;
      while (i < n) {
        var nxt = sizes[i];
        if (units && (used + nxt > maxTokens || (maxUnits != null && units >= maxUnits))) break;
        if (units === 0 && nxt > maxTokens) {
          i += 1;
          units = 1;
          used = nxt;
          break;
        }
        used += nxt;
        units += 1;
        i += 1;
      }
      ranges.push([start, i]);
      if (i >= n) break;
      if (overlapTokens > 0 && i > start) {
        var acc = 0;
        var k = i;
        while (k > start && acc < overlapTokens) {
          k -= 1;
          acc += sizes[k];
        }
        if (k < i) i = k;
        if (i <= start) i = start + 1;
      } else if (overlapTokens === 0 && i === start) {
        i = start + 1;
      }
    }
    return ranges;
  }

  root.tokenWindows = tokenWindows;
  root.packUnitRanges = packUnitRanges;
})(typeof self !== "undefined" ? self : typeof global !== "undefined" ? global : window);
