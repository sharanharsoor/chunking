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

`check_js_fixtures.js` compares in-tab JS to the same `expected.json` for:

| Strategy | What must match |
|---|---|
| `fixed_size`, `sentence_based`, `overlapping_window` (characters, `preserve_boundaries: false`) | `start` / `end` / `content` (Unicode strings, not bytes; `💩` is one scalar) |
| `markdown_chunker` (`chunk_by=headers`, no preamble, no fenced `#`) | `content` after trim; Python does not set offsets |
| `csv_chunker` | `csv_start_row` / `csv_end_row` / `csv_row_count` (Python `csv.writer` uses `\r\n`) |
| `json_chunker` | `json_start_index` / `json_end_index` / `json_object_count` (Python re-dumps JSON) |
| `paragraph_based` | `paragraph_count` (`merge_short_paragraphs: false`; Python collapses whitespace) |
| `fixed_length_word` | `start_word_index` / `end_word_index` / `word_count`, plus `content` when spacing is a single space |
| `fastcdc` (gear) | UTF-8 byte ranges via `sha256`; JS `start_byte`/`end_byte` must cover the file |

This is not “the browser is byte-identical to pip.” Code strategies, `rolling_hash`, tiktoken, and embeddings stay unchecked here. Markdown with a preamble or `#` inside fences is lab JS behavior, not this golden. `fastcdc` compares UTF-8 byte ranges via `sha256` (gear, this library — not restic).
