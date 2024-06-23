from chat_modules.llamaindex.intergrations import IntergrationsChatModule
from memory_module import ChatMemory
from llama_index.core.storage.chat_store import SimpleChatStore
from llama_index.core.chat_engine import SimpleChatEngine
from chainlit.input_widget import Select, Switch, Slider
import chainlit as cl
from config import params
import time

# Define params
default_params = {
    "temperature": 0.7,
    "max_tokens": 512,
    "session_key": "user_12345"
}


def set_chat_engine(settings):
    global chat_engine
    # Chat module sample
    chat_module = IntergrationsChatModule(service_name = "GROQ",
                                          model_name = settings["model_name"],
                                          temperature = settings["temperature"])
    chat_model = chat_module.get_chat_model()

    # Define memory
    chat_store = SimpleChatStore()
    memory = ChatMemory.get_session_memory(chat_store = chat_store,
                                           session_key = default_params["session_key"],
                                           token_limit = settings["max_tokens"])

    # Define chat engine
    chat_engine = SimpleChatEngine(llm = chat_model,
                                   memory = memory,
                                   prefix_messages = [])

@cl.on_chat_start
async def start_chat():
    cl.user_session.set(
        "message_history",
        [{"role": "system", "content": "You are a helpful assistant."}],
    )

    models = params.llamaindex_services["GROQ"]["CHAT_MODELS"]
    settings = await cl.ChatSettings(
        [
            Select(
                id = "model_name",
                label = "Groq Model",
                values = models,
                initial_index = 0,
            ),
            Slider(
                id = "temperature",
                label = "Temperature",
                initial = default_params['temperature'],
                min = 0,
                max = 2,
                step=0.1,
            ),
            Slider(
                id = "max_tokens",
                label = "Max tokens",
                initial = default_params['max_tokens'],
                min = 128,
                max = 8192,
                step = 128,
            ),
        ]
    ).send()
    set_chat_engine(settings = settings)

@cl.on_settings_update
async def setup_agent(settings):
    # Chat module sample
    set_chat_engine(settings = settings)

@cl.on_message
async def main(message: cl.Message):

    message_history = cl.user_session.get("message_history")
    message_history.append({"role": "user", "content": message.content})

    msg = cl.Message(content="")
    await msg.send()

    # Get stream
    beginTime = time.time()
    stream = chat_engine.stream_chat(message.content)

    count = 0
    # Stream object
    for (i,part) in enumerate(stream.response_gen):
        count += 1
        await msg.stream_token(part)
    duration = time.time() - beginTime

    await msg.stream_token(f"\n( Response in {round(duration,2)}s with {round(count/duration,2)} tokens/s)")
    # Append to history
    message_history.append({"role": "assistant", "content": msg.content})
    await msg.update()
