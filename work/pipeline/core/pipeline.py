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
       - Saves intermediate results (optional)
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
        
        # Intermediate results
        self.answers: List[Answer] = []
        self.evaluation_results: Dict[str, List[Dict]] = {}
    
    def run(
        self,
        dataset_path: str,
        resume_from: Optional[str] = None,
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
        answers = self._generate_answers(questions, resume_from)
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
        
        # Print summaries
        for eval_name, result in evaluation_results.items():
            print(result.summary())
        
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
        resume_from: Optional[str] = None,
    ) -> List[Answer]:
        """Generate answers for all questions"""
        
        # Load existing answers if resuming
        existing_answers = {}
        if resume_from and Path(resume_from).exists():
            print(f"📥 Resuming from {resume_from}")
            with open(resume_from, 'r', encoding='utf-8') as f:
                for line in f:
                    ans = json.loads(line)
                    existing_answers[ans['question_id']] = Answer(
                        question_id=ans['question_id'],
                        answer_text=ans.get('hypothesis', ans.get('answer_text', '')),
                        confidence=ans.get('confidence'),
                        processing_time=ans.get('processing_time'),
                        metadata=ans.get('metadata', {}),
                    )
            print(f"📥 Loaded {len(existing_answers)} existing answers")
        
        answers = []
        
        # Progress bar
        pbar = tqdm(questions, desc="Generating answers")
        
        for question in pbar:
            # Skip if already answered
            if question.question_id in existing_answers:
                answers.append(existing_answers[question.question_id])
                continue
            
            pbar.set_description(f"Processing {question.question_id}")
            
            try:
                # Process context
                self.memory_system.process_context(question.context)
                
                # Generate answer
                start_time = time.time()
                answer = self.memory_system.answer_question(
                    question.question_text,
                    question.question_id
                )
                answer.processing_time = time.time() - start_time
                
                answers.append(answer)
                
                # Save intermediate result
                if self.config.save_intermediate:
                    self._save_intermediate_answer(answer)
                
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
                
                # Save intermediate evaluation results
                if self.config.save_intermediate:
                    self._save_intermediate_evaluation(eval_name, evaluation_result)
                
            except Exception as e:
                print(f"❌ Error in evaluator {eval_name}: {e}")
                import traceback
                traceback.print_exc()
        
        return results
    
    def _save_intermediate_answer(self, answer: Answer) -> None:
        """Save a single answer to intermediate results file"""
        output_file = self.output_dir / f"{self.config.experiment_name}_answers.jsonl"
        with open(output_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(answer.to_dict()) + '\n')
    
    def _save_intermediate_evaluation(
        self,
        evaluator_name: str,
        result: EvaluationResult
    ) -> None:
        """Save evaluation results to file"""
        safe_name = evaluator_name.replace(' ', '_').replace('/', '-')
        output_file = self.output_dir / f"{self.config.experiment_name}_{safe_name}_results.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result.to_dict(), f, indent=2)
    
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
