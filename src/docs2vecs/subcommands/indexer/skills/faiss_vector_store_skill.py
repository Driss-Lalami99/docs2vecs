from pathlib import Path
from typing import List
from typing import Dict
from typing import Optional
from typing import Any

import faiss

import numpy as np

from docs2vecs.subcommands.indexer.config.config import Config
from docs2vecs.subcommands.indexer.document.document import Document
from docs2vecs.subcommands.indexer.skills.skill import IndexerSkill
from docs2vecs.subcommands.indexer.skills.tracker import VectorStoreTracker


class FaissVectorStoreSkill(IndexerSkill):
    """
    Faiss vector store skill for storing and retrieving document embeddings.
    Supports both flat (exact) and approximate nearest neighbor search.
    """

    def __init__(self, config: Dict[str, Any], global_config: Config , vector_store_tracker: Optional[VectorStoreTracker] = None) -> None:
        super().__init__(config, global_config)
        self._vector_store_tracker = vector_store_tracker
      
    def run(self, input: Optional[List[Document]] = None) -> List[Document]:
        self.logger.info("Running FaissVectorStoreSkill...")
        print("Hello World !")
        db_path = Path(self._config["db_path"]).expanduser().resolve().as_posix()
        # faiss_client = self._get_client(db_path)
    #     faiss_collection = faiss_client.get_or_create_collection(self._config["collection_name"])

    #     self.logger.debug(f"Going to process {len(input)} documents")
    #     for doc in input:
    #         self.logger.debug(f"Processing document: {doc.filename}")
    #         faiss_collection.upsert(
    #             ids=[chunk.chunk_id for chunk in doc.chunks],
    #             embeddings=[chunk.embedding for chunk in doc.chunks],
    #             documents=[chunk.content for chunk in doc.chunks],
    #             metadatas=[{"source": chunk.source_link, "tags": doc.tag} for chunk in doc.chunks],
    #         )

    #     return input


    # def faiss_index(self, dimenison: int) -> None:
    #     print("Hello World !")
    #     index = faiss.IndexFlatL2(dimenison)  
    #     self.logger(index.is_trained)


    