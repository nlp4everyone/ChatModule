from dotenv import load_dotenv
import os,json
load_dotenv()
cache_folder = "cache_folder"
embedding_model_folder = os.path.join(cache_folder,"embedding_model")

# Embedding model params
# Define params
embedding_service = "COHERE"
embedding_model_name = "embed-english-light-v3.0"
chat_service = "GEMINI"
chat_model = "models/gemini-pro"

# Model config
AI21_KEY = os.getenv("AI21_KEY")
ANTHROPIC_KEY = os.getenv("ANTHROPIC_KEY") # Max usage 10$
CLARIFAI_KEY = os.getenv("CLARIFAI_KEY")
COHERE_KEY = os.getenv("COHERE_KEY") # Cohere limited calls per minutes
GRADIENT_KEY = os.getenv("GRADIENT_KEY") # Cohere limited calls per minutes
GROQ_KEY = os.getenv("GROQ_KEY") # 30 requests/min
KONKO_KEY = os.getenv("KONKO_KEY") # 5$ starter bundle
LLAMAAPI_KEY = os.getenv("LLAMAAPI_KEY") # 5$ starter bundle
OPENAI_KEY = os.getenv("OPENAI_KEY") # 5$ starter bundle
PERPLEXITY_KEY = os.getenv("PERPLEXITY_KEY") # Required payment
TOGETHER_KEY = os.getenv("TOGETHER_KEY") # 25$ starter bundle
GEMINI_KEY = os.getenv("GEMINI_KEY") # Free to use
VOYAGE_KEY = os.getenv("VOYAGE_KEY") # Free for 50M first tokens
NOMIC_KEY = os.getenv("NOMIC_KEY") # Free for 50M first tokens
LLAMAPARSE_KEY = os.getenv("LLAMAPARSE_KEY") # https://cloud.llamaindex.ai/parse

# Load service
if not os.path.exists("config/llama_index_config.json"):
    raise Exception("Llama index config is not existed!")

# Load config
with open("config/llama_index_config.json",'r') as f:
    llamaindex_services = json.load(f)

# Add key service
llamaindex_services["AI21"]["KEY"] = AI21_KEY
llamaindex_services["ANTHROPIC"]["KEY"] = ANTHROPIC_KEY # List models: https://docs.anthropic.com/claude/docs/models-overview
llamaindex_services["CLARIFAI"]["KEY"] = CLARIFAI_KEY
llamaindex_services["COHERE"]["KEY"] = COHERE_KEY # List models: https://docs.cohere.com/docs/command-beta  List embbeding: https://docs.cohere.com/reference/embed
llamaindex_services["GRADIENT"]["KEY"] = GRADIENT_KEY
llamaindex_services["GROQ"]["KEY"] = GROQ_KEY
llamaindex_services["OPENAI"]["KEY"] = OPENAI_KEY # List model: https://platform.openai.com/docs/models/continuous-model-upgrades
llamaindex_services["PERPLEXITY"]["KEY"] = PERPLEXITY_KEY # List model: https://docs.perplexity.ai/docs/model-cards
llamaindex_services["TOGETHER"]["KEY"] = TOGETHER_KEY  # List model: https://docs.together.ai/docs/inference-models
llamaindex_services["GEMINI"]["KEY"] = GEMINI_KEY # List model: https://ai.google.dev/models/gemini
llamaindex_services["QDRANT"]["KEY"] = ""  # Qdrant Embedding: https://qdrant.github.io/fastembed/examples/Supported_Models/
llamaindex_services["VOYAGE"]["KEY"] = VOYAGE_KEY  # Embedding: https://docs.voyageai.com/docs/pricing
llamaindex_services["NOMIC"]["KEY"] = NOMIC_KEY # Embedding: https://docs.nomic.ai/atlas/models/text-embedding
