/* fastcdc, hash_algorithm=gear. Port of chunking_strategy.strategies.general.fastcdc_chunker. Offsets are UTF-8 bytes; highlighter gets Unicode scalars. */
(function (root) {
  var GEAR = (function () {
    var table = [];
    for (var i = 0; i < 256; i++) {
      var value = i;
      for (var k = 0; k < 8; k++) {
        if (value & 1) value = (value >>> 1) ^ 0xEDB88320;
        else value >>>= 1;
      }
      table.push(value >>> 0);
    }
    return table;
  })();

  function byteToScalar(decoder, bytes, byteOffset) {
    if (byteOffset <= 0) return 0;
    if (byteOffset >= bytes.length) return Array.from(decoder.decode(bytes)).length;
    var prefix = decoder.decode(bytes.subarray(0, byteOffset)).replace(/\uFFFD+$/, "");
    return Array.from(prefix).length;
  }

  function initGear(data, start, minPos) {
    var hashVal = 0;
    var end = Math.min(minPos, data.length);
    for (var i = start; i < end; i++) {
      hashVal = (hashVal * 2 + GEAR[data[i]]) >>> 0;
    }
    return hashVal;
  }

  function updateGear(hashVal, data, pos) {
    if (pos >= data.length) return hashVal;
    return (hashVal * 2 + GEAR[data[pos]]) >>> 0;
  }

  function isBoundary(hashVal, mask) {
    return (hashVal & mask) === 0;
  }

  function findBoundary(data, start, minSize, maxSize, avgSize, maskBits, normalization) {
    var dataLen = data.length;
    var minPos = Math.min(start + minSize, dataLen);
    var maxPos = Math.min(start + maxSize, dataLen);
    if (minPos >= dataLen) return dataLen;
    var hashVal = initGear(data, start, minPos);
    var pos = minPos;
    var mask = (1 << maskBits) - 1;
    if (normalization) {
      var normalSize = avgSize;
      var backupPos = null;
      var backupMask = (1 << (maskBits - 1)) - 1;
      var limit = Math.min(start + normalSize * 2, maxPos);
      while (pos < limit) {
        if (isBoundary(hashVal, mask)) return pos;
        if (pos >= start + normalSize && backupPos === null && (hashVal & backupMask) === 0) {
          backupPos = pos;
        }
        if (pos < dataLen - 1) {
          hashVal = updateGear(hashVal, data, pos);
          pos += 1;
        } else break;
      }
      if (backupPos !== null) return backupPos;
    } else {
      while (pos < maxPos) {
        if (isBoundary(hashVal, mask)) return pos;
        if (pos < dataLen - 1) {
          hashVal = updateGear(hashVal, data, pos);
          pos += 1;
        } else break;
      }
    }
    return maxPos;
  }

  function chunkFastCdc(text, params) {
    var minSize = Math.max(1, Number(params.min_chunk_size) || 32);
    var avgSize = Math.max(minSize, Number(params.avg_chunk_size) || 64);
    var maxSize = Math.max(avgSize, Number(params.max_chunk_size) || 128);
    var maskBits = Math.max(1, Number(params.mask_bits) || 6);
    var normalization = params.normalization !== false;
    var extra = {
      chunker_used: "fastcdc",
      source: params.source || "paste",
      algorithm: "gear",
    };
    if (!text) return [];
    var bytes = new TextEncoder().encode(text);
    var decoder = new TextDecoder("utf-8");
    var chunks = [];
    var chunkStart = 0;
    while (chunkStart < bytes.length) {
      var chunkEnd = findBoundary(bytes, chunkStart, minSize, maxSize, avgSize, maskBits, normalization);
      var meta = Object.assign({}, extra, {
        start_byte: chunkStart,
        end_byte: chunkEnd,
        chunk_size: chunkEnd - chunkStart,
      });
      chunks.push({
        id: "fastcdc_" + chunks.length,
        content: decoder.decode(bytes.subarray(chunkStart, chunkEnd)),
        start: byteToScalar(decoder, bytes, chunkStart),
        end: byteToScalar(decoder, bytes, chunkEnd),
        size: chunkEnd - chunkStart,
        offset_unit: "char",
        metadata: meta,
      });
      chunkStart = chunkEnd;
    }
    return chunks;
  }

  root.chunkFastCdc = chunkFastCdc;
})(typeof self !== "undefined" ? self : typeof global !== "undefined" ? global : window);
