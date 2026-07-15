# docs2vecs Samples

This directory contains end-to-end samples that demonstrate how to use the `docs2vecs` indexer with local files.

## Available Samples

| Sample | Purpose |
|--------|---------|
| [PDF only](./pdf-only/) | Index a local PDF file into a generated ChromaDB collection |
| [Markdown only](./markdown-only/) | Index a local Markdown file into a generated ChromaDB collection |

## Prerequisites

- Python 3.11 or higher
- `uv`
- A checkout of this repository

## Usage Pattern

Run sample commands from the repository root:

```bash
uv run docs2vecs indexer --config samples/<sample-name>/config.yml
```

Each sample writes a local ChromaDB database under its own `output/` directory. Those generated files are not committed to the repository; they are created when you run the sample.

The first run may take longer while the local embedding model is downloaded.