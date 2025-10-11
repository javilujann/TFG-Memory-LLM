#!/usr/bin/env python3
"""
Full Context Baseline Test
- Loads questions and chat history from LongMemEval Oracle dataset
- Feeds all context to Ollama model
- Saves answers in JSONL format for evaluation
- Runs the evaluation script
"""

import json
import os
import ollama
import time
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def load_ollama_config():
    """Load Ollama configuration from .env file"""
    return {
        'host': os.getenv('OLLAMA_HOST', 'http://localhost:11434'),
        'model': os.getenv('OLLAMA_MODEL', 'qwen3:32b'),  # Using your model from test_ollama.py
        'timeout': int(os.getenv('OLLAMA_TIMEOUT', '120'))
    }

def query_ollama(prompt, config):
    """Send a prompt to Ollama and get the response using native library"""
    try:
        print(f"🤖 Querying Ollama model: {config['model']}")
        
        # Create Ollama client
        client = ollama.Client(host=config['host'])
        
        # Generate response
        response = client.generate(
            model=config['model'],
            prompt=prompt,
            options={
                'temperature': 0.1,  # Low temperature for consistent answers
                'top_p': 0.9,
            }
        )
        
        return response['response'].strip()
        
    except Exception as e:
        print(f"❌ Error querying Ollama: {e}")
        return f"Error: Could not get response from Ollama - {e}"

def format_chat_history(sessions):
    """Format chat history sessions into a readable context"""
    context = "=== CHAT HISTORY ===\n\n"
    
    for i, session in enumerate(sessions, 1):
        context += f"Session {i}:\n"
        for turn in session:
            role = turn.get('role', 'unknown')
            content = turn.get('content', '')
            if role == 'user':
                context += f"User: {content}\n"
            elif role == 'assistant':
                context += f"Assistant: {content}\n"
        context += "\n"
    
    return context

def create_full_context_prompt(question, chat_history):
    """Create a prompt with full chat history context"""
    context = format_chat_history(chat_history)
    
    prompt = f"""{context}

=== QUESTION ===
Based on the chat history above, please answer the following question:

{question}

=== INSTRUCTIONS ===
- Use only information from the chat history above
- Be specific and accurate
- If you cannot find the answer in the chat history, say "I don't have enough information to answer this question"
- Keep your answer concise and direct

Answer:"""
    
    return prompt

def process_dataset(dataset_path, output_path, config, max_questions=None):
    """Process the oracle dataset and generate answers"""
    
    print(f"📂 Loading dataset: {dataset_path}")
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if max_questions:
        data = data[:max_questions]
        print(f"🔢 Limiting to first {max_questions} questions for testing")
    
    print(f"📊 Processing {len(data)} questions...")
    
    results = []
    
    for i, entry in enumerate(data, 1):
        question_id = entry['question_id']
        question = entry['question']
        chat_history = entry['haystack_sessions']
        
        print(f"\n🔍 Question {i}/{len(data)}: {question_id}")
        print(f"📝 Question: {question[:100]}{'...' if len(question) > 100 else ''}")
        
        # Create prompt with full context
        prompt = create_full_context_prompt(question, chat_history)
        
        # Get answer from Ollama
        answer = query_ollama(prompt, config)
        
        # Save result in the format expected by evaluation script
        result = {
            "question_id": question_id,
            "hypothesis": answer
        }
        results.append(result)
        
        print(f"✅ Answer: {answer[:100]}{'...' if len(answer) > 100 else ''}")
        
        # Small delay to avoid overwhelming Ollama
        time.sleep(0.5)
    
    # Save results to JSONL file
    print(f"\n💾 Saving results to: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')
    
    print(f"✅ Saved {len(results)} answers")
    return output_path

def run_evaluation(answers_file, dataset_file, evaluator_model='qwen3-32b'):
    """Run the LongMemEval evaluation script"""
    
    print(f"\n🧪 Running evaluation...")
    print(f"📊 Answers file: {answers_file}")
    print(f"📚 Dataset file: {dataset_file}")
    print(f"🤖 Evaluator model: {evaluator_model}")
    
    import subprocess
    import sys
    
    # Path to the evaluation script
    eval_script = Path("src/evaluate_qa.py")
    
    if not eval_script.exists():
        print(f"❌ Evaluation script not found: {eval_script}")
        return False
    
    try:
        # Run the evaluation script
        cmd = [
            sys.executable,
            str(eval_script),
            evaluator_model,
            answers_file,
            dataset_file
        ]
        
        print(f"🚀 Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        print("\n" + "="*60)
        print("📊 EVALUATION RESULTS")
        print("="*60)
        print(result.stdout)
        
        if result.stderr:
            print("\n⚠️ Evaluation warnings/errors:")
            print(result.stderr)
        
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        print("❌ Evaluation timed out")
        return False
    except Exception as e:
        print(f"❌ Error running evaluation: {e}")
        return False

def main():
    """Main function to run the full context baseline test"""
    
    print("🚀 LongMemEval Full Context Baseline Test")
    
    # Configuration
    config = load_ollama_config()
    dataset_file = "data/longmemeval_oracle.json"
    answers_file = "answers/full_context_answers.jsonl"
    
    print(f"🔧 Configuration:")
    print(f"   Ollama Host: {config['host']}")
    print(f"   Model: {config['model']}")
    print(f"   Dataset: {dataset_file}")
    print(f"   Output: {answers_file}")
    
    # Check if dataset exists
    if not Path(dataset_file).exists():
        print(f"❌ Dataset file not found: {dataset_file}")
        return
    
    # Test Ollama connection
    try:
        test_response = query_ollama("Hello, this is a test.", config)
        if test_response.startswith("Error:"):
            print(f"❌ Cannot connect to Ollama: {test_response}")
            return
        print(f"✅ Ollama connection successful")
    except Exception as e:
        print(f"❌ Ollama connection failed: {e}")
        return
    
    # Process dataset (start with just 5 questions for testing)
    max_questions = 5  # Change this to None to process all questions
    
    try:
        output_file = process_dataset(
            dataset_file, 
            answers_file, 
            config, 
            max_questions=max_questions
        )
        
        # Run evaluation
        success = run_evaluation(output_file, dataset_file, evaluator_model="gpt-5-nano")
        
        if success:
            print("\n🎉 Full context baseline test completed successfully!")
        else:
            print("\n❌ Evaluation failed")
            
    except KeyboardInterrupt:
        print("\n⏹️ Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")

if __name__ == "__main__":
    main()