from chat_modules.llamaindex import ServiceChatModule
from embedding_modules.llamaindex import ServiceEmbeddingModule

# chat_module = ServiceChatModule()
# print(chat_module.chat("Hello"))

embedding_module = ServiceEmbeddingModule()
embedding_model = embedding_module.get_embedding_model()
print(embedding_model.get_text_embedding("Hello"))