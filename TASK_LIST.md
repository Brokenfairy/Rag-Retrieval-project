# Task List

- [x] Harden `StateManager` with thread-safe and atomic JSON writes.
- [x] Fix pipeline orchestration status flow and metadata handling.
- [x] Save uploads before background processing in Streamlit app.
- [x] Cache heavy runtime objects (`InvoicePipeline`, `AnalystAgent`) for smoother UX.
- [x] Optimize indexing with batched embeddings and dynamic vector dimension detection.
- [x] Add resilient fallbacks for parser/converter error paths.
- [x] Improve Milvus search-tool readiness/error handling.
- [x] Improve ingestion robustness for non-UTF8 text files.
- [x] Add missing dependency (`crewai-tools`) to `requirements.txt`.
- [x] Run syntax validation checks (`compileall`) after changes.

- [x] Install/update dependencies in active runtime environment.
- [x] Run full end-to-end validation with real invoice files and Milvus backend.
