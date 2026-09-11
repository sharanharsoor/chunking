/* csv_chunker, chunk_by=rows. Quoted fields + quoted newlines. Offsets are original row spans. */
(function (root) {
  function parseCsv(text, delimiter) {
    var records = [];
    var row = [];
    var field = "";
    var rowStart = 0;
    var inQuotes = false;
    var i = 0;

    function endRow(end) {
      row.push(field);
      field = "";
      records.push({ start: rowStart, end: end, cells: row });
      row = [];
    }

    for (i = 0; i < text.length; i++) {
      var ch = text[i];
      if (inQuotes) {
        if (ch === '"') {
          if (text[i + 1] === '"') {
            field += '"';
            i += 1;
          } else {
            inQuotes = false;
          }
        } else {
          field += ch;
        }
        continue;
      }
      if (ch === '"') {
        inQuotes = true;
        continue;
      }
      if (ch === delimiter) {
        row.push(field);
        field = "";
        continue;
      }
      if (ch === "\n" || ch === "\r") {
        endRow(i);
        if (ch === "\r" && text[i + 1] === "\n") i += 1;
        rowStart = i + 1;
        continue;
      }
      field += ch;
    }
    if (field.length || row.length || (text.length && rowStart < text.length)) {
      endRow(text.length);
    }
    return records;
  }

  function guessDelimiter(text) {
    var nl = text.search(/\r?\n/);
    var header = nl < 0 ? text : text.slice(0, nl);
    var commas = (header.match(/,/g) || []).length;
    var tabs = (header.match(/\t/g) || []).length;
    var semis = (header.match(/;/g) || []).length;
    if (tabs > commas && tabs >= semis) return "\t";
    if (semis > commas && semis >= tabs) return ";";
    return ",";
  }

  function isEmptyRow(cells) {
    for (var i = 0; i < cells.length; i++) {
      if (String(cells[i]).trim()) return false;
    }
    return true;
  }

  function chunkCsv(text, params) {
    var rowsPer = Math.max(1, Number(params.rows_per_chunk) || 1000);
    var preserveHeaders = params.preserve_headers !== false;
    var extra = { chunker_used: "csv_chunker", source: params.source || "paste", chunk_by: "rows" };
    if (!text) return [];
    var records = parseCsv(text, params.delimiter || guessDelimiter(text));
    if (params.skip_empty_lines !== false) {
      records = records.filter(function (r) { return !isEmptyRow(r.cells); });
    }
    if (!records.length) {
      return [root.chunkFromUtf16("csv_chunker_0", text, 0, text.length, extra)];
    }
    var header = records[0];
    var data = records.slice(1);
    if (preserveHeaders) extra.csv_header = text.slice(header.start, header.end);
    if (!data.length) {
      return [root.chunkFromUtf16("csv_chunker_0", text, header.start, header.end, extra)];
    }
    var chunks = [];
    for (var i = 0; i < data.length; i += rowsPer) {
      var group = data.slice(i, i + rowsPer);
      var start = preserveHeaders && i === 0 ? header.start : group[0].start;
      var end = group[group.length - 1].end;
      var meta = Object.assign({}, extra, {
        csv_row_count: group.length,
        csv_start_row: i + 1,
        csv_end_row: i + group.length,
      });
      chunks.push(root.chunkFromUtf16("csv_chunker_" + chunks.length, text, start, end, meta));
    }
    return chunks;
  }

  root.chunkCsv = chunkCsv;
})(typeof self !== "undefined" ? self : typeof global !== "undefined" ? global : window);
