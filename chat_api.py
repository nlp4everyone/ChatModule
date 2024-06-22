from memory_module import ChatMemory
from chat_modules.llamaindex.intergrations import IntergrationsChatModule
from llama_index.core.chat_engine import SimpleChatEngine
from llama_index.storage.chat_store.redis import RedisChatStore
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel
import os

# Default params
session_key = "user_12345"
# Define chat storing service
chat_store = RedisChatStore(redis_url = "redis://redis-chat-database:6379")
# Define memory
chat_memory = ChatMemory.get_session_memory(chat_store = chat_store, session_key = session_key)

class InputQuestion(BaseModel):
    question: str = ""
    user_id : str = session_key
    temperature :float = 0.8
    max_tokens :int = 512

openai_tags = [
    {"name": "Server Ping"},
    {"name": "Chat Response","description":"Generation text from input question"},
    {"name": "Users","description":"Get information about user"},
    {"name": "Users Message","description":"Action alongside users message"}
]
# Define FastAPI
app = FastAPI(openapi_tags = openai_tags)

def check_key_existed(user_id :str):
    # Get all user id includes inside DB
    user_ids = chat_store.get_keys()

    # Check whether user id existed or not
    if user_id not in user_ids:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail="User ID is not existed!")

@app.get("/", tags = ["Server Ping"])
async def get_server_response():
    return {"message": "Server is on!"}

@app.post('/chat/generate',tags = ["Chat Response"])
async def get_answer(input : InputQuestion):
    # Get answer 
    if len(input.question) == 0:
        raise HTTPException(status_code = status.HTTP_412_PRECONDITION_FAILED,
                            detail = "Question cannot be empty!")

    # Get llm
    chat_module = IntergrationsChatModule(temperature = input.temperature, max_tokens = input.max_tokens)
    llm = chat_module.get_chat_model()

    # Get chat engine
    chat_engine = SimpleChatEngine.from_defaults(memory=chat_memory, llm=llm)
    answer = await chat_engine.achat(input.question)
    return {"user_id":input.user_id, "question": input.question, "answer": answer.response}

@app.get('/user/ids/all', tags = ["Users"])
async def get_user_ids():
    # Get all user id includes inside DB
    user_ids = chat_store.get_keys()
    return {'user_ids': user_ids}

@app.get('/messages/{user_id}/get', tags = ["Users Message"])
async def get_user_messages(user_id :str):
    # Check key existed
    check_key_existed(user_id = user_id)

    # Get messages
    try:
        messages = chat_store.get_messages(key = user_id)
        return {"user_id": user_id, "messages": messages}
    except:
        return {"status": f"Get messages from user id {user_id} failed"}

@app.delete('/messages/{user_id}/delete/all', tags=["Users Message"])
async def delete_all_user_messages(user_id: str):
    # Check key existed
    check_key_existed(user_id=user_id)

    try:
        # Get messages
        messages = chat_store.delete_messages(key=user_id)
        return {"user_id": user_id, "messages": messages}
    except:
        return {"status": f"Delete message with user id {user_id} failed"}

@app.delete('/messages/{user_id}/delete/last', tags=["Users Message"])
async def delete_last_user_message(user_id: str):
    # Check key existed
    check_key_existed(user_id=user_id)

    try:
        # Get messages
        messages = chat_store.delete_last_message(key=user_id)
        return {"user_id": user_id, "messages": messages}
    except:
        return {"status": f"Delete message with user id {user_id} failed"}