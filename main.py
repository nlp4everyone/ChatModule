from ai_modules.chatmodel_modules import ServiceChatModel

service_chatmodel = ServiceChatModel()
res = service_chatmodel.chat("Hello")
print(res)