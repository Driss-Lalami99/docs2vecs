from pathlib import Path
from typing import List
from typing import Dict
from typing import Optional
from typing import Any

import faiss
import os
from docs2vecs.subcommands.indexer.config.config import Config
from docs2vecs.subcommands.indexer.document.document import Document
from docs2vecs.subcommands.indexer.skills.skill import IndexerSkill
from docs2vecs.subcommands.indexer.skills.tracker import VectorStoreTracker
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain_community.docstore.in_memory import InMemoryDocstore


class FaissVectorStoreSkill(IndexerSkill):
    """
    Faiss vector store skill for storing and retrieving document embeddings.
    Supports both flat (exact) and approximate nearest neighbor search.
    """

    def __init__(self, config: Dict[str, Any], global_config: Config , vector_store_tracker: Optional[VectorStoreTracker] = None) -> None:
        super().__init__(config, global_config)
        self._vector_store_tracker = vector_store_tracker     

    def run(self, input: Optional[List[Document]] = None) -> List[Document]:
        VECTOR_DIMENSION = self._config["dimension"]
        self.logger.info("Running FaissVectorStoreSkill...")
        db_path = Path(self._config["db_path"]).expanduser().resolve().as_posix()
        faiss_index = faiss.IndexFlatL2(VECTOR_DIMENSION)   
        for doc in input :
            self.logger.debug(f"Processing document: {doc.filename}")
            ids=[chunk.chunk_id for chunk in doc.chunks]
            embeddings=[chunk.embedding for chunk in doc.chunks]
            documents=[chunk.content for chunk in doc.chunks]
            metadatas=[{"source": chunk.source_link, "tags": doc.tag} for chunk in doc.chunks]
            # Check if the FAISS vector store already exists

        if os.path.exists(db_path):
            self.logger.info(f"Existing FAISS vector store found at {db_path}, loading and updating it.")
            vector_store = FAISS.load_local(db_path, embeddings=self.get_embeddings(input), allow_dangerous_deserialization=True)
            # Get existing IDs as a set
            existing_ids = set(vector_store.index_to_docstore_id.values())
        else:
            self.logger.info(f"No existing FAISS vector store found at {db_path}, creating a new one.")
            vector_store = FAISS(
                index=faiss_index,
                embedding_function=self.get_embeddings(input),
                docstore=InMemoryDocstore(),
                index_to_docstore_id={}
            )
            existing_ids = set()

        # Filter out duplicates
        filtered_ids = [id for id in ids if id not in existing_ids]
        filtered_embeddings = [embeddings[i] for i, id_ in enumerate(ids) if id_ not in existing_ids]
        filtered_documents = [documents[i] for i, id_ in enumerate(ids) if id_ not in existing_ids]
        filtered_metadatas = [metadatas[i] for i, id_ in enumerate(ids) if id_ not in existing_ids]

        if filtered_ids:
            vector_store.add_embeddings(
                text_embeddings=zip(filtered_documents, filtered_embeddings),
                metadatas=filtered_metadatas,
                ids=filtered_ids
            )
        else:
            self.logger.info("No new embeddings to add (all ids already exist).")

        vector_store.save_local(db_path)
        self.logger.info(f"FAISS vector store saved at {db_path}")

        return input

    def get_embeddings(self, input: Optional[List[Document]] = None) -> List[float]:
        data = []
        for doc in input:
           self.logger.debug(f"Processing document: {doc.filename}")
           for chunk in doc.chunks:
                data.append(chunk.embedding)
        return data
