from llama_index.core.llms import LLM

class StandardlizedChatModule():
    def __init__(self,
                 temperature: float = 0.8,
                 max_tokens :int = 512):
        """Define general method for chatmodel service"""

        # Default params
        self.temperature = temperature
        self.max_tokens = max_tokens
        # Define history
        self.history = []

        # Default model
        self._chat_model = None


    def get_chat_model(self) -> LLM:
        # Return chat model
        return self._chat_model
