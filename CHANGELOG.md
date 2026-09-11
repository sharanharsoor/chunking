# Changelog

## Unreleased (0.5.0)

### Changed
- Default `sentence_based` splitter is `simple_v1` (the previous `simple` splitter is still available by name).
- `[text]` extra no longer pulls `sentence-transformers` / torch. Use `[ml]` for embeddings.
- Core classifiers include Windows. `[media]`, `[tika]`, and `[hardware]` stay POSIX-oriented.
- Recommended install is `chunking-strategy` or `chunking-strategy[tiktoken]`. `[all]` is not recommended.

### Added
- `MissingExtraError` with a one-line pip hint (`[tiktoken]`, `[ml]`, `[tika]`).
- Result-level `quality_score`: 0.4 size consistency + 0.4 sentence-boundary + 0.2 span coverage. CDC / code / JSON skip the `.!?` term.
- `chunking-strategy compare FILE -s a,b,c` table (n_chunks, avg_size, quality_score, elapsed).
- `chunking-strategy list-strategies --tier lab|python_only|all`.
- `[pdf]` and `[media]` extras (`[media]` matches `[multimedia]`).
