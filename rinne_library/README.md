# Local Rinne library data

The public project may include generic library code, but it does not publish a
user's Rinne library contents.

Documents, images, indexes, generated metadata and other library data placed in
this directory are local-only and ignored by Git. Public tests and examples
must use synthetic data outside this directory.

At first run the backend creates `rinne_01/documents`, `rinne_01/images` and
`rinne_01/videos`, each with a `待整理` inbox. Upload through `/library/upload`
or copy supported files into those folders. The `rinne-library` MCP server can
list folders, search filenames, read documents and pass a selected image back
to the configured dialogue model.

Literal filename search works without extra packages. Semantic filename search
is optional and stays offline: install `sentence-transformers`, place
`BAAI/bge-base-zh-v1.5` in the local Hugging Face cache, or set
`RINNE_LIBRARY_MODEL_CACHE` and `RINNE_LIBRARY_EMBEDDING_MODEL`. If unavailable,
the MCP server falls back to literal search and never downloads a model.
