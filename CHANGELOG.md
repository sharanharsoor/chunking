# Changelog

## Unreleased

### Added
- Docs site at `poc/docs/` (GitHub Pages `/docs/`): lab, install, strategies, N-way compare, bring-your-own, Python API, CLI, formats. Linked from the playground.
- Lab toolbox points N-way compare and custom strategies at pip, with links into those docs.
- Lab hierarchical `recursive`: parent/child tree (paragraph then sentence), matched to the Python golden. This is not `recursive_character`.
- Lab `python_code` uses tree-sitter for module-level `def`/`class` (indent fallback). pip stays stdlib `ast`; nested methods are separate chunks. Compared on `symbol_name` / `symbol_kind` / `line_start` / `line_end`.
- Lab `semantic` loads Transformers.js MiniLM from Hugging Face (~25 MB) on first pick. Grey until that load finishes. Cuts are not claimed to match pip `[ml]`.

### Changed
- Lab tab icon is a two-column mark (copper / teal) instead of the browser default globe.
- Lab sidebar width can be dragged so the source and A/B lists get more of the screen.

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
