#!/usr/bin/env python3
"""
Simple test script for Ollama library
"""

import ollama
import time
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Create Ollama client with host from environment variable or default
ollama_host = os.getenv('OLLAMA_HOST', 'http://localhost:11434')
client = ollama.Client(host=ollama_host)

def test_ollama_connection():
    """Test basic connection to Ollama"""
    try:
        print(f"🔗 Connecting to Ollama at: {ollama_host}")
        
        listResponse = client.list()
        models = listResponse.models
        print("✅ Connection successful!")
        print(f"Available models: {len(models)}")
        for model in models:
            print(f"  - {model.model}")
        return True
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False

def test_simple_generation(model_name="llama3.2"):
    """Test simple text generation"""
    try:
        print(f"\n🧠 Testing generation with {model_name}...")
        
        prompt = "What is the capital of France?"
        start_time = time.time()
        
        response = client.generate(
            model=model_name,
            prompt=prompt,
            options={
                'temperature': 0.7,
                'max_tokens': 100
            }
        )
        
        end_time = time.time()
        
        print(f"✅ Generation successful!")
        print(f"Prompt: {prompt}")
        print(f"Response: {response['response']}")
        print(f"Time taken: {end_time - start_time:.2f} seconds")
        print(f"Tokens: {response.get('eval_count', 'N/A')}")
        
        return True
    except Exception as e:
        print(f"❌ Generation failed: {e}")
        return False

def test_chat_format(model_name="llama3.2"):
    """Test chat-style conversation"""
    try:
        print(f"\n💬 Testing chat format with {model_name}...")
        
        messages = [
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': 'Explain AI in one sentence.'}
        ]
        
        response = client.chat(
            model=model_name,
            messages=messages
        )
        
        print(f"✅ Chat successful!")
        print(f"Response: {response['message']['content']}")
        
        return True
    except Exception as e:
        print(f"❌ Chat failed: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 Starting Ollama library tests...\n")
    
    # Test 1: Connection
    if not test_ollama_connection():
        print("\n❌ Ollama not running or not accessible. Please start Ollama first.")
        return
    
    # Test 2: Simple generation
    test_simple_generation("qwen3:32b")
    
    # Test 3: Chat format
    test_chat_format("qwen3:32b")

    print("\n✅ All tests completed!")

if __name__ == "__main__":
    main()