from typing import Literal, Union
from config.params import *
# from llama_index.embeddings.nomic import NomicEmbedding
from ai_modules.embedding_modules.base_embedding import BaseEmbeddingTemplate
from system_components import Logger

class ServiceEmbedding(BaseEmbeddingTemplate):
    def __init__(self,
                 model_name: str = "default",
                 service_name: Union[Literal["COHERE","GRADIENT","MISTRAL","OPENAI","TOGETHER","VOYAGE","NOMIC"],str] = "COHERE",
                 batch_size: int = 10,
                 max_length : int = 1024):
        """Define embedding service with specified params"""

        super().__init__(batch_size = batch_size,max_length= max_length)
        # Define variables
        list_services = list(supported_services.keys())
        # Check service available
        if service_name not in list_services:
            Logger.exception(f"Service {service_name} is not supported!")
            raise Exception(f"Service {service_name} is not supported!")

        self._model_name = model_name
        # Get model name
        if model_name == "default":
            # List models
            list_models = supported_services[service_name]["EMBBEDDING_MODELS"]
            # Check list
            if not isinstance(list_models,list) or len(list_models) == 0:
                raise Exception(f"Wrong list of models")
            # Take first element
            self._model_name = list_models[0]
            # Check name
            if len(self._model_name) == 0: raise Exception("Model name cant be empty")

        # Define key
        self._api_key = supported_services[service_name]["KEY"]
        # TOGETHER service
        if service_name == "TOGETHER":
            from llama_index.embeddings.together import TogetherEmbedding
            self._embedding_model = TogetherEmbedding(model_name = self._model_name,
                                                      api_key = self._api_key)

        elif service_name == "COHERE":
            from llama_index.embeddings.cohere import CohereEmbedding
            self._embedding_model = CohereEmbedding(model_name = self._model_name,
                                                    cohere_api_key = self._api_key,
                                                    embed_batch_size = self.batch_size)

        elif service_name == "VOYAGE":
            from llama_index.embeddings.voyageai import VoyageEmbedding
            self._embedding_model = VoyageEmbedding(model_name = self._model_name,
                                                    voyage_api_key = self._api_key,
                                                    embed_batch_size = self.batch_size)

        elif service_name == "OPENAI":
            from llama_index.embeddings.openai import OpenAIEmbedding
            self._embedding_model = OpenAIEmbedding(model = self._model_name,
                                                    api_key = self._api_key,
                                                    embed_batch_size = self.batch_size)
        elif service_name == "MISTRAL":
            Logger.exception("Mistral currently required charge")
            raise Exception("Mistral currently required charge")

        # elif service_name == "NOMIC":
        #     self._embedding_model = NomicEmbedding(api_key=self.api_key,embed_batch_size=self.batch_size)
        else:
            service_exception_msg = f"Service {service_name} is not supported!"
            Logger.exception(service_exception_msg)
            raise Exception(service_exception_msg)

        #Logging Info
        Logger.info(f"Launch {service_name} service with embedding model {self._model_name}!")

