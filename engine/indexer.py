import os
import json
import faiss
from sentence_transformers import SentenceTransformer

class Indexer:
    """Handles the creation of the searchable FAISS index."""
    def __init__(self, sbert_model: SentenceTransformer):
        self.model = sbert_model
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        self.metadata = {}

    def build_index(self, corpus_path: str):
        """Builds the search index from all .txt files in a directory."""
        print(f"\nBuilding the search index from documents in '{corpus_path}'...")
        doc_files = [f for f in os.listdir(corpus_path) if f.endswith(".txt")]
        all_texts = []
        for doc_id, filename in enumerate(doc_files):
            with open(os.path.join(corpus_path, filename), "r", encoding='utf-8') as f:
                all_texts.append(f.read())
            self.metadata[doc_id] = filename
        
        print("  - Generating embeddings for all documents...")
        embeddings = self.model.encode(all_texts, show_progress_bar=True)
        self.index.add(embeddings)
        print("✅ Index built successfully.")

    def save(self, index_path="corpus.faiss", meta_path="metadata.json"):
        """Saves the index and metadata to disk."""
        print(f"  - Saving index to '{index_path}' and metadata to '{meta_path}'...")
        faiss.write_index(self.index, index_path)
        with open(meta_path, "w") as f:
            json.dump(self.metadata, f)
        print("✅ Index saved.")