/* rolling_hash. Polynomial CDC. Lab hashes UTF-8 bytes like pip; highlighter uses Unicode scalars. */
(function (root) {
  var MOD = 1000000007n;
  var BASE = 256n;

  function powMod(base, exp, mod) {
    var b = BigInt(base);
    var e = BigInt(exp);
    var m = BigInt(mod);
    var r = 1n;
    b %= m;
    while (e > 0n) {
      if (e & 1n) r = (r * b) % m;
      b = (b * b) % m;
      e >>= 1n;
    }
    return r;
  }

  function byteToScalar(decoder, bytes, byteOffset) {
    if (byteOffset <= 0) return 0;
    if (byteOffset >= bytes.length) {
      return Array.from(decoder.decode(bytes)).length;
    }
    var prefix = decoder.decode(bytes.subarray(0, byteOffset));
    prefix = prefix.replace(/\uFFFD+$/, "");
    return Array.from(prefix).length;
  }

  function chunkRollingHash(text, params) {
    var minSize = Math.max(1, Number(params.min_chunk_size) || 24);
    var target = Math.max(minSize, Number(params.target_chunk_size) || 64);
    var maxSize = Math.max(target, Number(params.max_chunk_size) || 120);
    var windowSize = Math.max(2, Number(params.window_size) || 8);
    var extra = { chunker_used: "rolling_hash", source: params.source || "paste", hash_function: "polynomial" };
    if (!text) return [];
    var Encoder = typeof TextEncoder !== "undefined" ? TextEncoder : null;
    var Decoder = typeof TextDecoder !== "undefined" ? TextDecoder : null;
    if (!Encoder || !Decoder) {
      return [root.chunkFromUtf16("rolling_hash_0", text, 0, text.length, extra)];
    }
    var bytes = new Encoder().encode(text);
    var decoder = new Decoder("utf-8");
    if (bytes.length <= minSize) {
      return [root.chunkFromUtf16("rolling_hash_0", text, 0, text.length, extra)];
    }
    var threshold = Math.max(1, Math.floor((1 << 20) / target));
    var basePower = powMod(BASE, windowSize - 1, MOD);
    var hash = 0n;
    var window = [];
    var chunks = [];
    var chunkStart = 0;
    var position = 0;
    function emit(endByte) {
      chunks.push({
        id: "rolling_hash_" + chunks.length,
        content: decoder.decode(bytes.subarray(chunkStart, endByte)),
        start: byteToScalar(decoder, bytes, chunkStart),
        end: byteToScalar(decoder, bytes, endByte),
        size: endByte - chunkStart,
        offset_unit: "char",
        metadata: extra,
      });
    }
    while (position < bytes.length) {
      var byteIn = bytes[position];
      window.push(byteIn);
      if (window.length > windowSize) {
        var byteOut = window.shift();
        hash = (((hash - BigInt(byteOut) * basePower) % MOD + MOD) % MOD * BASE + BigInt(byteIn)) % MOD;
      } else {
        hash = (hash * BASE + BigInt(byteIn)) % MOD;
      }
      var chunkSize = position - chunkStart + 1;
      var boundary = Number(hash % BigInt(threshold)) === 0 && chunkSize >= minSize;
      if (boundary || chunkSize >= maxSize) {
        emit(position + 1);
        chunkStart = position + 1;
      }
      position += 1;
    }
    if (chunkStart < bytes.length) emit(bytes.length);
    return chunks;
  }

  root.chunkRollingHash = chunkRollingHash;
})(typeof self !== "undefined" ? self : typeof global !== "undefined" ? global : window);
