# from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from config import params
from typing import Optional
from strenum import StrEnum
# from llama_index.embeddings.fastembed import FastEmbedEmbedding,base
from system_components import Logger
from embedding_modules.llamaindex.base_embedding_module import StandardlizedEmbeddingModule
import os

class EmbeddingProvider(StrEnum):
    HuggingFace = "HuggingFace",
    FastEmbed = "FastEmbed",


class ServingEmbeddingModule(StandardlizedEmbeddingModule):
    def __init__(self,
                 model_name: Optional[str] = None,
                 service_name: EmbeddingProvider = EmbeddingProvider.FastEmbed,
                 batch_size: int = 10,
                 max_length: int = 1024,
                 embedding_model_folder = params.embedding_model_folder):
        """Define embedding service with specified params"""

        super().__init__(batch_size = batch_size,max_length= max_length)
        # Define variable
        self._embedding_model_folder = embedding_model_folder
        self.model_name = model_name

        # Create folder
        os.makedirs(embedding_model_folder,exist_ok=True)

        service_unsupported_msg = "HuggingFace temporally turned off"
        # Hugging Face
        if service_name == EmbeddingProvider.HuggingFace:
            # self._embedding_model = HuggingFaceEmbedding(cache_folder=self._embedding_model_folder,embed_batch_size=self.batch_size)
            Logger.exception(service_unsupported_msg)
            raise Exception(service_unsupported_msg)
        # Fast Embed
        elif service_name == EmbeddingProvider.FastEmbed:
            service_unsupported_msg = "FastEmbed temporally turned off"
            Logger.exception(service_unsupported_msg)
            raise Exception(service_unsupported_msg)
        else:
            service_unsupported_msg = f"Service {service_name} is not supported!"
            Logger.exception(service_unsupported_msg)
            raise Exception(service_unsupported_msg)

        # Check model name
        if self.model_name is not None: self._embedding_model.model_name = model_name
        # Logging status
        init_message = f"Initiate {service_name} with model: {self._embedding_model.model_name}, batch size {self.batch_size}"
        Logger.info(init_message)


