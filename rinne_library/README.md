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

## Video observation

Video observation is a separate, optional sidecar. Static images continue to
go directly to the selected dialogue model. To enable videos:

1. Install `ffmpeg` and `ffprobe` and make both commands available on `PATH`.
2. Put only the Gemini API key in the Git-ignored file
   `local_config/gemini_video_api_key.txt`.
3. Set `character_config.agent_config.media_analysis.enabled` to `True`.

`POST /library/upload` accepts `video_mode=normal` (default), which creates a
space-saving 720p H.264/AAC local copy, or `video_mode=original`, which keeps a
validated copy of the source. Long videos are split without re-encoding before
being sent to the observer. Observations are cached under ignored library data
and are invalidated when the video or the user's current question changes.

Environment variables `RINNE_MEDIA_GEMINI_API_KEY`,
`RINNE_MEDIA_GEMINI_BASE_URL`, and `RINNE_MEDIA_GEMINI_MODEL` can override the
file and YAML values. Secrets must not be committed.
