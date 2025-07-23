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
    Supports flat search using IndexFlatL2 
    """

    def __init__(self, config: Dict[str, Any], global_config: Config , vector_store_tracker: Optional[VectorStoreTracker] = None) -> None:
        super().__init__(config, global_config)
        self._vector_store_tracker = vector_store_tracker   
        self._overwrite_index = self._config.get("overwrite_index", False)  
        self.VECTOR_DIMENSION = self._config.get("dimension")
        self._faiss_index = faiss.IndexFlatL2(self.VECTOR_DIMENSION)

    def run(self, input: Optional[List[Document]] = None) -> List[Document]:
        self.logger.info("Running FaissVectorStoreSkill...")
        db_path = Path(self._config["db_path"]).expanduser().resolve().as_posix()

        for doc in input:
            self.logger.info(f"Processing document: {doc.filename}")
            ids=[chunk.chunk_id for chunk in doc.chunks]
            embeddings=[chunk.embedding for chunk in doc.chunks]
            documents=[chunk.content for chunk in doc.chunks]
            metadatas=[{"source": chunk.source_link, "tags": doc.tag} for chunk in doc.chunks]
            # Check if the FAISS vector store already exists
            self.logger.info(f"total number of chunks: {len(doc.chunks)}")

            if os.path.exists(db_path):
                self.logger.info(f"Existing FAISS vector store found at {db_path}, loading and updating it.")
                vector_store = FAISS.load_local(db_path, embeddings=self._get_embeddings(input), allow_dangerous_deserialization=True)
                
                if self._overwrite_index :
                    self._cleanup_index(vector_store, ids)

            else:
                self.logger.info(f"No existing FAISS vector store found at {db_path}, creating a new one.")
                vector_store = FAISS(
                    index=self._faiss_index,
                    embedding_function=self._get_embeddings(input),
                    docstore=InMemoryDocstore(),
                    index_to_docstore_id={}
                )

            # Add new embeddings
            if self._overwrite_index :
                vector_store.add_embeddings(
                    text_embeddings=zip(documents, embeddings),
                    metadatas=metadatas,
                    ids=ids
                )
            else:
                self.logger.debug(f"list of ids to add: {ids}")
                self.logger.debug(f"values of docstore : {list(vector_store.index_to_docstore_id.values())}")
                for id in ids:
                    if id in list(vector_store.index_to_docstore_id.values()):
                        self.logger.debug(f"ID {id} already exists in the index, skipping.")
                    else:
                        self.logger.debug(f"ID {id} does not exist in the index, adding it.")
                        vector_store.add_embeddings(
                            text_embeddings=[(documents[ids.index(id)], embeddings[ids.index(id)])],
                            metadatas=[metadatas[ids.index(id)]],
                            ids=[id]
                        )
            # Save the vector store
            vector_store.save_local(db_path)
            self.logger.info(f"FAISS vector store saved at {db_path} with {len(vector_store.index_to_docstore_id)}")

        return input

    def _get_embeddings(self, input: Optional[List[Document]] = None) -> List[float]:
        data = []
        for doc in input:
           self.logger.debug(f"Processing document: {doc.filename}")
           for chunk in doc.chunks:
                data.append(chunk.embedding)
        return data

    def _cleanup_index(self, vector_store : FAISS, ids: List[str]) :
        existing_ids = set(vector_store.index_to_docstore_id.values())
        ids_to_remove = []
        for id in ids:
            if id in existing_ids :
                self.logger.info(f"Found existing ID to remove: {id}")
                ids_to_remove.append(id)
        if len(ids_to_remove) > 0:
            self.logger.debug(f"the ids to remove : {ids_to_remove}")
            self.logger.debug(f"the current index mapping : {vector_store.index_to_docstore_id}")
            vector_store.delete(ids=ids_to_remove)
        self.logger.info(f"the index mapping after deletion : {vector_store.index_to_docstore_id}")