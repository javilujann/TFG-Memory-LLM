from dotenv import load_dotenv
import ollama
from mem0 import Memory, MemoryClient
import os

from pydantic import config
from typing import Any

# Load environment variables
load_dotenv()

# Create Ollama client
ollama_host = os.getenv('OLLAMA_HOST', 'http://localhost:11434')
ollamaClient = ollama.Client(host=ollama_host)
answerModel = os.getenv('OLLAMA_MODEL', 'llama3.3:latest')

# Create Memory client
def create_memory_client(ollama_host: str) -> MemoryClient | Memory:
    useLocal = os.getenv('MEM0_USE_LOCAL_OLLAMA', False)     
    
    if not useLocal:
        api_key = os.getenv("MEM0_API_KEY")
        return MemoryClient(api_key=api_key)
    
    model = os.getenv('MEMORY_MODEL', 'llama3.3:latest')

    config = {
        "llm": {
            "provider": "ollama",
            "config": {
                "model": model,
                "temperature": 0,
                "max_tokens": 2000,
                "ollama_base_url": ollama_host,
            },
        },
        "embedder": {
            "provider": "ollama",
            "config": {
                "model": "nomic-embed-text:latest",
                "ollama_base_url": ollama_host,
            },
        },
        "vector_store": {
            "provider": "supabase",
            "config": {
                "connection_string": os.getenv("SUPABASE_URL"),
                "collection_name": "memories",
                "embedding_model_dims": 768,
            },
        },
    }
    return Memory.from_config(config)

memClient = create_memory_client(ollama_host)

# Mem0 call agnostic to client or local instance
def search_memories(query: str, filters: dict) -> Any:
    if isinstance(memClient, Memory):
        # Extract user_id filter for local instance
        for filter in filters.get("OR", []):
            if "user_id" in filter:
                user_id = filter["user_id"]
                del filter["user_id"]

        return memClient.search(query=query,user_id=user_id)
    
    else:
        return memClient.search(query=query, version="v2", filters=filters)
    

def add_memory(messages: list, user_id: str) -> None:
    if isinstance(memClient, Memory):
        memClient.add(messages=messages, user_id=user_id)
    else:
        memClient.add(messages=messages, user_id=user_id, version="v2")

def retrieve_memory_string(memories: Any) -> str:
    if isinstance(memClient, Memory):
        return "\n".join(f"- {mem['memory']}" for mem in memories['results'])
    else:
        return "\n".join(f"- {mem['memory']}" for mem in memories)


# Test Ollama connection
def test_ollama_connection():
    """Test basic connection to Ollama"""
    try:
        print(f"🔗 Connecting to Ollama at: {ollama_host}")
        
        listResponse = ollamaClient.list()
        models = listResponse.models
        print("✅ Connection successful!")
        print(f"Available models: {len(models)}")

        return True
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False

# Chat with memories
def chat_with_memories(message:str, user_id:str) -> str:

    # Retrieve recent memories
    filters = {
        "OR":[
            {
                "user_id":user_id
            }
        ]
    }
    memories = search_memories(query=message, filters=filters)
    print(memories)
    memoriesStr = retrieve_memory_string(memories)

    # Create system prompt with memories
    systemPrompt = f"You are a helpful AI assistant that answers the user question using the following context:\n{memoriesStr}\n\n"
    messages = [{"role": "system", "content": systemPrompt}, {"role": "user", "content": message}]

    # Get response from Ollama
    response = ollamaClient.chat(model=answerModel, messages=messages)
    responseStr = response['message']['content']

    # Save the new interaction to memory
    messages.append({"role": "assistant", "content": responseStr})
    add_memory(messages, user_id=user_id)

    return responseStr

def main():
    if not test_ollama_connection():
        return
    
    user_id = "user123"
    while True:
        user_message = input("You: ").strip()
        if user_message.lower() in ["exit", "quit"]:
            break
        response = chat_with_memories(user_message, user_id)
        print(f"AI: {response}\n")

if __name__ == "__main__":
    main()