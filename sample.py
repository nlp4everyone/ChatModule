from ai_modules.chatmodel_modules import ServiceChatModel
from ai_modules.embedding_modules import ServiceEmbedding

chat_service = ServiceChatModel(service_name="ANTHROPIC")
ans = chat_service.chat("Hello")
print(ans)