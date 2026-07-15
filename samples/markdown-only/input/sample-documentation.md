# Getting Started with Documentation

This sample Markdown document demonstrates how the `docs2vecs` indexer processes local Markdown files.

## Overview

Documentation often contains concepts, procedures, and examples that are useful in retrieval augmented generation workflows. Indexing the content into a vector store makes it searchable by semantic meaning.

## Key Concepts

### Vector Embeddings

Vector embeddings are numerical representations of text. They help compare pieces of text by meaning instead of exact wording.

### Document Chunking

Long documents are split into smaller chunks before embedding. Smaller chunks make retrieval more precise and keep each vector focused on a coherent idea.

### Local Vector Store

This sample stores generated vectors in a local ChromaDB database under the sample output directory.

## Example Workflow

1. Scan the input directory for Markdown files.
2. Read Markdown content from each file.
3. Split the text into chunks.
4. Generate local embeddings.
5. Store the chunks and embeddings in ChromaDB.

## Conclusion

This sample is intentionally small so new users can inspect the input, configuration, command, and generated output in one place.