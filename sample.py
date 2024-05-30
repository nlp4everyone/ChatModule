from ai_modules.chatmodel_modules import ServiceChatModel
from ai_modules.embedding_modules import ServiceEmbedding
from llama_index.core.llms import ChatMessage
import asyncio

embedding_service = ServiceEmbedding(service_name="COHERE")
embedding_model = embedding_service.get_embedding_model()