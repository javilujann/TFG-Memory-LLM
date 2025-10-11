from dotenv import load_dotenv
import ollama
from mem0 import MemoryClient
import os

# Load environment variables
load_dotenv()
memClient = MemoryClient(api_key= os.getenv("MEM0_API_KEY"))

ollama_host = os.getenv('OLLAMA_HOST', 'http://localhost:11434')
ollamaClient = ollama.Client(host=ollama_host)

model = os.getenv('OLLAMA_MODEL', 'llama3.3:latest')

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

def chat_with_memories(message:str, user_id:str) -> str:

    # Retrieve recent memories
    filters = {
        "OR":[
            {
                "user_id":user_id
            }
        ]
    }
    memories = memClient.search(query=message, version="v2", filters=filters)
    memoriesStr = "\n".join(f"- {mem['memory']}" for mem in memories)

    # Create system prompt with memories
    systemPrompt = f"You are a helpful AI assistant that answers the user question using the following context:\n{memoriesStr}\n\n"
    messages = [{"role": "system", "content": systemPrompt}, {"role": "user", "content": message}]

    # Get response from Ollama
    response = ollamaClient.chat(model=model, messages=messages)
    responseStr = response['message']['content']

    # Save the new interaction to memory
    messages.append({"role": "assistant", "content": responseStr})
    memClient.add(messages, user_id=user_id, version="v2")

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