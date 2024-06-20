from llama_index.core.memory import ChatMemoryBuffer,ChatSummaryMemoryBuffer
from typing import Literal, Optional
from llama_index.core.memory.types import BaseMemory
from llama_index.core.storage.chat_store.base import BaseChatStore

class ChatMemory():

    @staticmethod
    def get_session_memory(chat_store: BaseChatStore,
                        session_key: str,
                        token_limit: int = 3000,
                        summary_prompt: Optional[str] = None,
                        memory_type: Literal["default", "summary"] = "default") -> BaseMemory:
        """Set chat memory type (Chat Memory Buffer, Chat Summary Memory Buffer) from Redis Storing"""
        assert session_key, "Session key cant be none"

        # Default memory type
        if memory_type == "default":
            chat_memory = ChatMemoryBuffer.from_defaults(
                token_limit = token_limit,
                chat_store = chat_store,
                chat_store_key = session_key
            )
        else:
            chat_memory = ChatSummaryMemoryBuffer.from_defaults(
                token_limit = token_limit,
                chat_store = chat_store,
                chat_store_key = session_key,
                summarize_prompt = summary_prompt
            )
        return chat_memory