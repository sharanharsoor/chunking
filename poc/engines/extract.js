/* Extract PDF (PDF.js) and Word (mammoth) to plain text, then the lab chunks that text.
   Engines load on demand. Extractors may differ from pip; chunkers match on the same text. */
(function (root) {
  var PDF_SRC = "engines/pdfjs/pdf.min.js";
  var PDF_WORKER = "engines/pdfjs/pdf.worker.min.js";
  var MAMMOTH_SRC = "engines/mammoth.browser.min.js";
  var pdfPromise = null;
  var mammothPromise = null;

  function sniffKind(name, buf) {
    var u8 = buf instanceof Uint8Array ? buf : new Uint8Array(buf);
    var ext = String(name || "").toLowerCase().match(/\.([a-z0-9]+)$/);
    ext = ext ? ext[1] : "";
    if (u8.length >= 4 && u8[0] === 0x25 && u8[1] === 0x50 && u8[2] === 0x44 && u8[3] === 0x46) return "pdf";
    if (u8.length >= 8 && u8[0] === 0xD0 && u8[1] === 0xCF && u8[2] === 0x11 && u8[3] === 0xE0) return "ole";
    if (u8.length >= 3 && u8[0] === 0xFF && u8[1] === 0xD8 && u8[2] === 0xFF) return "media";
    if (u8.length >= 4 && u8[0] === 0x89 && u8[1] === 0x50 && u8[2] === 0x4E && u8[3] === 0x47) return "media";
    if (u8.length >= 3 && u8[0] === 0x47 && u8[1] === 0x49 && u8[2] === 0x46) return "media";
    if (u8.length >= 2 && u8[0] === 0x50 && u8[1] === 0x4B) {
      var sample = latin1(u8, Math.min(u8.length, 262144));
      if (sample.indexOf("word/document") !== -1) return "docx";
      if (sample.indexOf("xl/workbook") !== -1) return "xlsx";
      if (sample.indexOf("ppt/slides") !== -1) return "pptx";
      if (ext === "docx") return "docx";
      if (ext === "xlsx") return "xlsx";
      if (ext === "pptx") return "pptx";
      return "zip";
    }
    if (ext === "pdf") return "pdf";
    if (ext === "docx") return "docx";
    if (ext === "doc" || ext === "rtf" || ext === "odt") return "ole";
    if (ext === "xlsx" || ext === "xls") return "xlsx";
    if (ext === "pptx" || ext === "ppt") return "pptx";
    if (/^(png|jpe?g|gif|webp|svg|mp3|mp4|wav|m4a|mov|avi)$/.test(ext)) return "media";
    return "text";
  }

  function latin1(u8, n) {
    var out = "";
    for (var i = 0; i < n; i++) out += String.fromCharCode(u8[i]);
    return out;
  }

  function itemX(it) { return it.transform[4]; }
  function itemY(it) { return it.transform[5]; }

  function itemsToText(items) {
    var rows = [];
    for (var i = 0; i < items.length; i++) {
      var it = items[i];
      if (!it || !it.str) continue;
      rows.push(it);
    }
    rows.sort(function (a, b) {
      var dy = itemY(b) - itemY(a);
      if (Math.abs(dy) > 3) return dy;
      return itemX(a) - itemX(b);
    });
    var lines = [];
    var line = "";
    var lastY = null;
    var lastEnd = 0;
    for (var r = 0; r < rows.length; r++) {
      var cur = rows[r];
      var y = itemY(cur);
      var x = itemX(cur);
      if (lastY != null && Math.abs(y - lastY) > 4) {
        lines.push(line.replace(/\s+$/, ""));
        line = "";
        lastEnd = 0;
      }
      var bit = cur.str;
      if (line && x - lastEnd > 1.5 && !/ $/.test(line) && !/^\s/.test(bit)) line += " ";
      line += bit;
      lastEnd = x + (cur.width || 0);
      lastY = y;
    }
    if (line) lines.push(line.replace(/\s+$/, ""));
    return lines.join("\n");
  }

  function pageHasColumns(items, pageWidth) {
    if (!items || items.length < 12 || !(pageWidth > 0)) return false;
    var mid = pageWidth / 2;
    var leftMax = -Infinity;
    var rightMin = Infinity;
    var leftN = 0;
    var rightN = 0;
    for (var i = 0; i < items.length; i++) {
      var it = items[i];
      if (!it || !it.str) continue;
      var x = itemX(it);
      if (x < mid - 8) {
        leftMax = Math.max(leftMax, x);
        leftN++;
      } else if (x > mid + 8) {
        rightMin = Math.min(rightMin, x);
        rightN++;
      }
    }
    if (leftN < 4 || rightN < 4) return false;
    return (rightMin - leftMax) > pageWidth * 0.12;
  }

  function loadScript(src) {
    return new Promise(function (resolve, reject) {
      if (typeof document === "undefined") {
        reject(new Error("ENGINE"));
        return;
      }
      var found = document.querySelector('script[data-engine="' + src + '"]');
      if (found) {
        resolve();
        return;
      }
      var s = document.createElement("script");
      s.src = src;
      s.setAttribute("data-engine", src);
      s.onload = function () { resolve(); };
      s.onerror = function () { reject(new Error("ENGINE")); };
      document.head.appendChild(s);
    });
  }

  function loadPdfJs() {
    if (pdfPromise) return pdfPromise;
    pdfPromise = loadScript(PDF_SRC).then(function () {
      var lib = root.pdfjsLib;
      if (!lib) throw new Error("ENGINE");
      lib.GlobalWorkerOptions.workerSrc = new URL(PDF_WORKER, root.location.href).href;
      return lib;
    }).catch(function (err) {
      pdfPromise = null;
      throw err;
    });
    return pdfPromise;
  }

  function loadMammoth() {
    if (mammothPromise) return mammothPromise;
    mammothPromise = loadScript(MAMMOTH_SRC).then(function () {
      if (!root.mammoth) throw new Error("ENGINE");
      return root.mammoth;
    }).catch(function (err) {
      mammothPromise = null;
      throw err;
    });
    return mammothPromise;
  }

  function extractPdf(lib, buf) {
    var data = buf instanceof Uint8Array ? buf : new Uint8Array(buf);
    return lib.getDocument({ data: data }).promise.then(function (doc) {
      var pages = [];
      var columns = false;
      var next = Promise.resolve();
      var n = doc.numPages;
      for (var p = 1; p <= n; p++) {
        next = next.then(function (pageNo) {
          return function () {
            return doc.getPage(pageNo).then(function (page) {
              var width = page.getViewport({ scale: 1 }).width;
              return page.getTextContent().then(function (content) {
                var items = content.items || [];
                if (pageHasColumns(items, width)) columns = true;
                pages.push(itemsToText(items));
              });
            });
          };
        }(p));
      }
      return next.then(function () {
        if (doc.destroy) doc.destroy();
        var text = pages.join("\n\n").replace(/[ \t]+\n/g, "\n").replace(/\n{3,}/g, "\n\n").trim();
        return { text: text, pages: n, columns: columns };
      });
    });
  }

  function extractDocx(mammoth, buf) {
    return mammoth.extractRawText({ arrayBuffer: buf }).then(function (result) {
      var text = String(result && result.value ? result.value : "").replace(/\n{3,}/g, "\n\n").trim();
      return { text: text };
    });
  }

  function pipOnlyMessage(kind) {
    if (kind === "xlsx") return "Spreadsheets stay in pip. Export CSV and drop that, or pip install chunking-strategy.";
    if (kind === "pptx") return "Slides stay in pip. pip install chunking-strategy.";
    if (kind === "ole") return "Old .doc and other Office binaries stay in pip. Save as .docx, or pip install chunking-strategy.";
    if (kind === "zip") return "That zip is not a Word file this tab can read. pip install chunking-strategy.";
    if (kind === "media") return "Images and media stay in pip. pip install chunking-strategy.";
    return "This preview extracts PDF and Word text. For that file type, pip install chunking-strategy.";
  }

  var api = {
    sniffKind: sniffKind,
    itemsToText: itemsToText,
    pageHasColumns: pageHasColumns,
    pipOnlyMessage: pipOnlyMessage,
    loadPdfJs: loadPdfJs,
    extractFromBuffer: function (name, buf) {
      var kind = sniffKind(name, buf);
      if (kind === "text") return Promise.resolve({ kind: "text", text: null });
      if (kind === "xlsx" || kind === "pptx" || kind === "ole" || kind === "zip" || kind === "media") {
        return Promise.reject({ code: "PIP_ONLY", kind: kind, message: pipOnlyMessage(kind) });
      }
      if (kind === "pdf") {
        return loadPdfJs().then(function (lib) {
          return extractPdf(lib, buf);
        }).then(function (out) {
          if (!out.text) {
            return Promise.reject({
              code: "EMPTY_PDF",
              kind: "pdf",
              message: "This PDF has no extractable text. Scanned pages need OCR in pip.",
            });
          }
          var note = "Text extracted with PDF.js. pip pypdf or PyMuPDF can differ on columns and layout.";
          if (out.columns) note = "This PDF may have columns. Text order might differ from the visual layout.";
          return { kind: "pdf", text: out.text, note: note, pages: out.pages };
        });
      }
      if (kind === "docx") {
        return loadMammoth().then(function (mammoth) {
          return extractDocx(mammoth, buf);
        }).then(function (out) {
          if (!out.text) {
            return Promise.reject({
              code: "EMPTY_DOCX",
              kind: "docx",
              message: "That Word file had no extractable text.",
            });
          }
          return {
            kind: "docx",
            text: out.text,
            note: "Text extracted with mammoth. pip python-docx can differ on formatting.",
          };
        });
      }
      return Promise.reject({ code: "PIP_ONLY", kind: kind, message: pipOnlyMessage(kind) });
    },
  };

  root.ChunkExtract = api;
  if (typeof module !== "undefined" && module.exports) module.exports = api;
})(typeof self !== "undefined" ? self : this);
