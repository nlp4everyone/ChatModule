from config.params import *
from typing import Union
# from llama_index.llms.gradient import GradientBaseModelLLM
from strenum import StrEnum
from chat_modules.llamaindex.base_chat_module import StandardlizedChatModule
from system_components import Logger

class ChatModelProvider(StrEnum):
    ANTHROPIC = "ANTHROPIC",
    COHERE = "COHERE",
    GRADIENT = "GRADIENT",
    GROQ = "GROQ",
    LLAMAAPI = "LLAMAAPI",
    OPENAI = "OPENAI",
    PERPLEXITY = "PERPLEXITY",
    TOGETHER = "TOGETHER",
    GEMINI = "GEMINI"

class IntergrationsChatModule(StandardlizedChatModule):
    def __init__(self,
                 model_name: str = "default",
                 service_name: Union[ChatModelProvider,str] = ChatModelProvider.GEMINI,
                 temperature: float = 0.8,
                 max_tokens :int = 512 ):
        """Define embedding service with specified params
        - model_name: str. Default is default
        - service_name: LLM Providers. Must be OpenAI, Anthoripic, etc
        - temperature:  a parameter that influences the language model's output, determining whether the output is more random and creative or more predictable
        """
        super().__init__(temperature = temperature,max_tokens = max_tokens)

        # Service support
        list_services = list(llamaindex_services.keys())
        # Check service available
        if service_name not in list_services:
            service_exception = f"Service {service_name} is not supported!"
            Logger.exception(service_exception)
            raise Exception(service_exception)

        # Define key
        self._api_key = llamaindex_services[service_name]["KEY"]

        # Default model
        self._chat_model = None

        self._model_name = model_name
        # Default model
        if model_name == "default":
            list_models = llamaindex_services[service_name]["CHAT_MODELS"]
            # Check type
            if not isinstance(list_models,list) or len(list_models) == 0:
                raise Exception(f"Wrong list of models")
            # Take first element
            self._model_name = list_models[0]
            # Check name
            if len(self._model_name) == 0: raise Exception("Model name cant be empty")


        # Other service
        if service_name == "ANTHROPIC":
            # Install dependency
            from llama_index.llms.anthropic import Anthropic
            self._chat_model = Anthropic(api_key = self._api_key,
                                         max_tokens = self.max_tokens,
                                         temperature = self.temperature,
                                         model = self._model_name)

        elif service_name == "COHERE":
            # Install dependency
            from llama_index.llms.cohere import Cohere
            self._chat_model = Cohere(api_key = self._api_key,
                                      max_tokens = self.max_tokens,
                                      temperature = self.temperature,
                                      model = self._model_name)

        elif service_name == "GRADIENT":
            raise Exception("Temporally not working")

        elif service_name == "GROQ":
            # Install dependency
            from llama_index.llms.groq import Groq
            self._chat_model = Groq(model = self._model_name,
                                    api_key = self._api_key)

        elif service_name == "LLAMAAPI":
            raise Exception("Temporally not working")
            # from llama_index.llms.llama_api import LlamaAPI
            # self._chat_model = LlamaAPI(temperature=self.temperature,max_tokens=self.max_tokens,api_key=self.api_key)

        elif service_name == "OPENAI":
            # Install dependency
            from llama_index.llms.openai import OpenAI
            self._chat_model = OpenAI(model = self._model_name,
                                      temperature = self.temperature,
                                      max_tokens = self.max_tokens,
                                      api_key = self._api_key,
                                      timeout = 15)

        elif service_name == "PERPLEXITY":
            # Install dependency
            from llama_index.llms.perplexity import Perplexity
            self._chat_model = Perplexity(model = self._model_name,
                                          temperature = self.temperature,
                                          max_tokens = self.max_tokens,
                                          api_key = self._api_key)

        elif service_name == "TOGETHER":
            # Install dependency
            from llama_index.llms.together import TogetherLLM
            self._chat_model = TogetherLLM(model = self._model_name,
                                           api_key = self._api_key,
                                           temperature = self.temperature)

        elif service_name == "GEMINI":
            # Install dependency
            from llama_index.llms.gemini import Gemini
            self._chat_model = Gemini(model_name = self._model_name,
                                      api_key = self._api_key,
                                      temperature = self.temperature,
                                      max_tokens = self.max_tokens)
        else:
            raise Exception(f"Service {service_name} is not supported!")


        # Print message
        Logger.info(f"Launch {service_name} service with chat model {self._model_name}, temperature of {self.temperature}")
