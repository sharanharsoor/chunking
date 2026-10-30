# Changelog

## Unreleased

## 0.6.0 (2026-10-30)

### Added
- Markdown heading packer: with `max_tokens` set, adjacent header sections pack until the next one would overflow the token budget instead of emitting a tiny heading-only chunk.
- Short header sections stick to a neighbor (under 250 characters, or under `max_tokens // 3` when packing) so RAG indexes do not vectorize orphan headings.
- Table parent windows on markdown/HTML table splits: each embedded slice can keep a wider `_parent_window` for retrieval excerpts.
- GFM table separator detection and safer HTML/markdown table piece joining.
- Lazy import of `discourse_aware_chunker` with the other heavy text strategies so importing the package does not hard-require that module at strategies package load time.
- Unit tests for the markdown packer, short-section merge, footer attach, and parent-window behavior.

### Fixed
- Oversized section after a short heading: the heading is prefixed onto the body before the token cut.
- Short tail after a full window is re-cut onto the previous window with overlap instead of being left as a sub-250 character orphan.
- Sections shorter than `min_chunk_size` merge into a neighbor rather than being dropped.
- Pages with no headings still take the same pack / short-merge pass.
- XML/HTML chunker table handling aligned with the markdown parent-window path.

## 0.5.0 (2026-09-11)

### Changed
- Default `sentence_based` splitter is `simple_v1` (the previous `simple` splitter is still available by name).
- `[text]` extra no longer pulls `sentence-transformers` / torch. Use `[ml]` for embeddings.
- Core classifiers include Windows. `[media]`, `[tika]`, and `[hardware]` stay POSIX-oriented.
- Recommended install is `chunking-strategy` or `chunking-strategy[tiktoken]`. `[all]` is not recommended.
- Package `__version__` matches `pyproject.toml` (it had stayed on 0.4.1 after 0.4.2).

### Added
- `MissingExtraError` with a one-line pip hint (`[tiktoken]`, `[ml]`, `[tika]`).
- Result-level `quality_score`: 0.4 size consistency + 0.4 sentence-boundary + 0.2 span coverage. CDC / code / JSON skip the `.!?` term.
- `chunking-strategy compare FILE -s a,b,c` table (n_chunks, avg_size, quality_score, elapsed).
- `chunking-strategy list-strategies --tier lab|python_only|all`.
- `[pdf]` and `[media]` extras (`[media]` matches `[multimedia]`).
- Windows CI smoke: core `pip install -e .` on `windows-latest`, then import, `fixed_size` / `sentence_based` chunk, and `list-strategies --tier lab`.
- Lab extracts PDF and Word in the browser (PDF.js + mammoth), then runs the same strategy names as text. Images, xlsx, pptx, and OLE `.doc` stay pip-only. There is no lab strategy named `pdf_chunker`.
