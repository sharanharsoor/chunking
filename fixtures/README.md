# Golden fixtures

One folder per (strategy, notable param set). Python is the first writer of `expected.json`.

```
fixtures/<strategy>/<case>/
  input.txt | input.bin
  params.json
  expected.json
```

Extractor tests (later) live under `fixtures/extract/` and are **not** required to match each other. Chunker tests that start from a checked-in extracted text **are**.

## Changing an algorithm

1. Change Python.
2. `python tools/run_fixture.py <folder> --write`
3. Review the `expected.json` diff in the PR (must be explainable).
4. Port TS to match.
5. CI `--check` enforces both runtimes.

Who regenerates: the person changing the algorithm. CI never `--write`.

```
python tools/run_fixture.py --check
python tools/run_fixture.py fixtures/sentence_based/simple_v1_unicode --write
node tools/check_js_fixtures.js
```

`check_js_fixtures.js` compares in-tab JS (`fixed_size`, `sentence_based`, `csv_chunker`) to the same `expected.json`. Text strategies must match `start` / `end` / `content` as Unicode strings (not bytes; `💩` is one scalar). CSV matches row spans (`csv_start_row`, `csv_end_row`, `csv_row_count`) because Python rewrites CSV text with `csv.writer` (`\r\n`).

This is not “the browser is byte-identical to pip.” It is three strategies, these five cases. Live knobs not in a fixture (e.g. `overlap_size > 0`) and the other in-tab chunkers are unchecked here. FastCDC / tiktoken / embeddings stay Python-only.
