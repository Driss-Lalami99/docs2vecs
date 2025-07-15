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
        db_path = Path(self._config["db_path"]).expanduser().resolve().as_posix()
        faiss_index = self.faiss_index()
        faiss.write_index(faiss_index, db_path)
        return input


    def faiss_index(self) :   
        index = faiss.IndexFlatL2(1000)   # build the index
        print(index.is_trained)                
        print(index.ntotal)
        return index

    