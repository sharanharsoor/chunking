/* MiniLM embeddings for the lab. Loaded as a module worker, never on first paint. */
import { pipeline, env } from "https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2";

env.allowLocalModels = false;
env.useBrowserCache = true;

var extractor = null;
var loading = null;

function warmup() {
  if (extractor) return Promise.resolve();
  if (loading) return loading;
  loading = pipeline("feature-extraction", "Xenova/all-MiniLM-L6-v2", { quantized: true })
    .then(function (pipe) {
      extractor = pipe;
      loading = null;
      return pipe;
    })
    .catch(function (err) {
      loading = null;
      throw err;
    });
  return loading;
}

function toVector(out) {
  if (!out) return [];
  if (out.data) return Array.from(out.data);
  if (Array.isArray(out)) return out;
  return Array.from(out);
}

function embedAll(texts) {
  // ponytail: one sentence at a time. MiniLM batching is the upgrade if a long doc stalls the tab.
  var i = 0;
  var vectors = [];
  function next() {
    if (i >= texts.length) return vectors;
    var t = texts[i];
    i += 1;
    return extractor(t, { pooling: "mean", normalize: true }).then(function (out) {
      vectors.push(toVector(out));
      return next();
    });
  }
  return warmup().then(next);
}

self.onmessage = function (ev) {
  var msg = ev.data || {};
  var id = msg.id;
  if (msg.type === "warmup") {
    warmup().then(function () {
      self.postMessage({ id: id, type: "ok" });
    }).catch(function (err) {
      self.postMessage({ id: id, type: "error", message: String(err && err.message ? err.message : err) });
    });
    return;
  }
  if (msg.type === "embed") {
    embedAll(msg.texts || []).then(function (vectors) {
      self.postMessage({ id: id, type: "ok", vectors: vectors });
    }).catch(function (err) {
      self.postMessage({ id: id, type: "error", message: String(err && err.message ? err.message : err) });
    });
  }
};
