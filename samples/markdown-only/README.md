# Markdown-Only Sample

This sample indexes a local Markdown document with the `docs2vecs indexer` pipeline.

## Files

| Path | Purpose |
|------|---------|
| `samples/markdown-only/input/sample-documentation.md` | Sample Markdown input document |
| `samples/markdown-only/config.yml` | Ready-to-run indexer configuration |
| `samples/markdown-only/output/` | Generated ChromaDB output directory created when the sample runs |

## Run

From the repository root:

```bash
uv run docs2vecs indexer --config samples/markdown-only/config.yml
```

## Expected Result

After the command completes, `samples/markdown-only/output/` should exist and contain a working ChromaDB database for the `markdown-sample-collection` collection.

You can verify the generated output by checking that:

- the command exits successfully,
- `samples/markdown-only/output/` is created,
- the ChromaDB files are present in that output directory.

Generated ChromaDB files are intentionally ignored by git.