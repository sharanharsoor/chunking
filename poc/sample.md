# Naive cuts miss structure

A heading should stay with the paragraph that explains it. Character windows do not care.

## Retrieval

RAG embeds each chunk. If you split on a fixed character count, this heading lands in a different window than the body.

## Takeaway

`markdown_chunker` cuts on `#` headers. `fixed_size` does not.
