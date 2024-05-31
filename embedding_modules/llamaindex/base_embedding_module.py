from llama_index.core.base.embeddings.base import BaseEmbedding

class StandardlizedEmbeddingModule():
    def __init__(self,batch_size: int = 10,max_length: int = 1024):
        super().__init__()
        """Define general method for embedding service"""
        # Define variable
        self.max_length = max_length
        self.batch_size = batch_size

        # Define embedding model
        self._embedding_model = None

    def get_embedding_model(self) -> BaseEmbedding:
        # Return embedding model
        return self._embedding_model