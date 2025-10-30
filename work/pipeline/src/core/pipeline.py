"""
Main Evaluation Pipeline Orchestrator

This module contains the EvaluationPipeline class that orchestrates
the entire evaluation flow, connecting all components together.
"""

import json
import time
from pathlib import Path
from typing import List, Optional, Dict, Any
from tqdm import tqdm

from .interfaces import DatasetReader, MemorySystem, Evaluator
from .models import Question, Answer, EvaluationResult, PipelineConfig


class EvaluationPipeline:
    """
    Main pipeline that orchestrates the evaluation flow.
    
    The pipeline:
    1. Loads questions from a dataset using a DatasetReader
    2. For each question:
       - Processes context with the MemorySystem
       - Gets an answer from the MemorySystem
       - Evaluates the answer with one or more Evaluators
    3. Aggregates and returns final results
    
    Example usage:
        pipeline = EvaluationPipeline(
            reader=LongMemEvalReader(),
            memory_system=FullContextMemorySystem(ollama_backend),
            evaluators=[LLMJudgeEvaluator()],
            config=config
        )
        results = pipeline.run('path/to/dataset.json')
    """
    
    def __init__(
        self,
        reader: DatasetReader,
        memory_system: MemorySystem,
        evaluators: List[Evaluator],
        config: PipelineConfig,
    ):
        """
        Initialize the evaluation pipeline.
        
        Args:
            reader: DatasetReader implementation to load questions
            memory_system: MemorySystem implementation to answer questions
            evaluators: List of Evaluator implementations to compute metrics
            config: PipelineConfig with all settings
        """
        self.reader = reader
        self.memory_system = memory_system
        self.evaluators = evaluators
        self.config = config
        
        # Output directory
        self.output_dir = Path(config.output_config.get('output_dir', 'outputs'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def run(
        self,
        dataset_path: str,
    ) -> Dict[str, EvaluationResult]:
        """
        Run the full evaluation pipeline.
        
        Args:
            dataset_path: Path to the dataset file
            resume_from: Path to intermediate results file to resume from (optional)
            
        Returns:
            Dictionary mapping evaluator names to their EvaluationResult objects
        """
        print(f"\n{'='*70}")
        print(f"🚀 Starting Evaluation Pipeline: {self.config.experiment_name}")
        print(f"{'='*70}\n")
        
        # Step 1: Load questions
        print("📂 Loading dataset...")
        questions = self._load_questions(dataset_path)
        print(f"✅ Loaded {len(questions)} questions")
        
        # Step 2: Generate answers
        print("\n🤖 Generating answers...")
        answers = self._generate_answers(questions)
        print(f"✅ Generated {len(answers)} answers")
        
        # Step 3: Evaluate answers
        print("\n📊 Evaluating answers...")
        evaluation_results = self._evaluate_answers(questions, answers)
        
        # Step 4: Save final results
        print("\n💾 Saving results...")
        self._save_final_results(evaluation_results)
        
        print(f"\n{'='*70}")
        print("✅ Pipeline completed successfully!")
        print(f"{'='*70}\n")
        
        return evaluation_results
    
    def _load_questions(self, dataset_path: str) -> List[Question]:
        """Load and validate questions from dataset"""
        questions = self.reader.load(
            dataset_path,
            max_questions=self.config.max_questions
        )
        
        # Validate questions
        valid_questions = []
        for q in questions:
            if self.reader.validate_question(q):
                valid_questions.append(q)
            else:
                print(f"⚠️  Warning: Invalid question {q.question_id}, skipping")
        
        if len(valid_questions) == 0:
            raise ValueError("No valid questions found in dataset")
        
        return valid_questions
    
    def _generate_answers(
        self,
        questions: List[Question],
    ) -> List[Answer]:
        """Generate answers for all questions"""
        # Progress bar
        pbar = tqdm(questions, desc="Generating answers")
        
        # Loop through questions and generate answers
        answers = []
        for question in pbar:
            pbar.set_description(f"Processing {question.question_id}")
            
            try:
                # Process context
                start_time = time.time()
                self.memory_system.process_context(question.context)
                context_time = time.time() - start_time

                # Generate answer
                start_time = time.time()
                answer = self.memory_system.answer_question(
                    question.question_text,
                    question.question_id
                )
                answer.processing_time = time.time() - start_time

                # Add context processing time and save answer
                answer.metadata['context_processing_time'] = context_time
                answers.append(answer)
                
                # Reset memory for next question
                self.memory_system.reset()
                
            except Exception as e:
                print(f"\n❌ Error processing question {question.question_id}: {e}")
                # Create error answer
                answer = Answer(
                    question_id=question.question_id,
                    answer_text=f"Error: {str(e)}",
                    metadata={'error': True, 'error_message': str(e)}
                )
                answers.append(answer)
        
        return answers
    
    def _evaluate_answers(
        self,
        questions: List[Question],
        answers: List[Answer],
    ) -> Dict[str, EvaluationResult]:
        """Evaluate answers with all evaluators"""
        
        results = {}
        
        for evaluator in self.evaluators:
            eval_name = evaluator.get_name()
            print(f"\n📊 Running evaluator: {eval_name}")
            
            try:
                # Evaluate each question
                per_question_results = []
                
                if evaluator.supports_batch_evaluation():
                    # Batch evaluation
                    per_question_results = evaluator.evaluate_batch(questions, answers)
                else:
                    # Individual evaluation
                    pbar = tqdm(zip(questions, answers), total=len(questions), desc=f"Evaluating with {eval_name}")
                    for question, answer in pbar:
                        result = evaluator.evaluate_single(question, answer)
                        per_question_results.append(result)
                
                # Aggregate results
                evaluation_result = evaluator.aggregate_results(
                    per_question_results,
                    questions
                )
                
                results[eval_name] = evaluation_result
                
            except Exception as e:
                print(f"❌ Error in evaluator {eval_name}: {e}")
                import traceback
                traceback.print_exc()
        
        return results
    
    
    def _save_final_results(self, results: Dict[str, EvaluationResult]) -> None:
        """Save final aggregated results"""
        output_file = self.output_dir / f"{self.config.experiment_name}_final_results.json"
        
        final_data = {
            'experiment_name': self.config.experiment_name,
            'config': self.config.to_dict(),
            'dataset_metadata': self.reader.get_metadata(),
            'memory_system': type(self.memory_system).__name__,
            'evaluators': {
                name: result.to_dict()
                for name, result in results.items()
            }
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(final_data, f, indent=2)
        
        print(f"💾 Final results saved to: {output_file}")
