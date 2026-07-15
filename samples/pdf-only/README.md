# PDF-Only Sample

This sample indexes a local PDF document with the `docs2vecs indexer` pipeline.

## Files

| Path | Purpose |
|------|---------|
| `samples/pdf-only/input/sample-guide.pdf` | Sample PDF input document |
| `samples/pdf-only/config.yml` | Ready-to-run indexer configuration |
| `samples/pdf-only/output/` | Generated ChromaDB output directory created when the sample runs |

## Run

From the repository root:

```bash
uv run docs2vecs indexer --config samples/pdf-only/config.yml
```

## Expected Result

After the command completes, `samples/pdf-only/output/` should exist and contain a working ChromaDB database for the `pdf-sample-collection` collection.

You can verify the generated output by checking that:

- the command exits successfully,
- `samples/pdf-only/output/` is created,
- the ChromaDB files are present in that output directory.

Generated ChromaDB files are intentionally ignored by git.