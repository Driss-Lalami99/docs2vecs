import pickle

def test_faiss_vector_store_skill():
  
    with open("C:\\Users\\dlalami\\Documents\\docs2vec_ressources\\faissdb_storage\\my_index.faiss\\index.pkl", "rb") as f:
        docstore, index_to_docstore_id = pickle.load(f)

    # Access the internal dictionary
    documents_dict = docstore._dict
    print(index_to_docstore_id)
    # Print all documents
    for doc_id, doc in documents_dict.items():
        print(f"ID: {doc_id}")
        print(f"Content: {doc.page_content}")
        print(f"Metadata: {doc.metadata}")
        print("-" * 40)

    assert len(documents_dict) == len(index_to_docstore_id)
    assert doc.page_content == "Hi my name is Driss Lalami, and I need to do better."
