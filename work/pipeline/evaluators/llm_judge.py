"""
LLM-as-Judge Evaluator

Uses an LLM (e.g., GPT-4) to evaluate answer correctness.
Based on the LongMemEval evaluation methodology.
"""

from typing import List, Dict, Any, Optional

from ..core.interfaces import Evaluator, LLMBackend
from ..core.models import Question, Answer, EvaluationResult


class LLMJudgeEvaluator(Evaluator):
    """
    Evaluator that uses an LLM to judge answer correctness.
    
    This is the approach used in LongMemEval and many QA benchmarks.
    Supports different evaluation prompts for different question types.
    """
    
    # Default prompt templates for different question types
    PROMPT_TEMPLATES = {
        'single-session-user': """I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no.

Question: {question}

Correct Answer: {answer}

Model Response: {response}

Is the model response correct? Answer yes or no only.""",
        
        'single-session-assistant': """I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no.

Question: {question}

Correct Answer: {answer}

Model Response: {response}

Is the model response correct? Answer yes or no only.""",
        
        'multi-session': """I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no.

Question: {question}

Correct Answer: {answer}

Model Response: {response}

Is the model response correct? Answer yes or no only.""",
        
        'temporal-reasoning': """I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. In addition, do not penalize off-by-one errors for the number of days. If the question asks for the number of days/weeks/months, etc., and the model makes off-by-one errors (e.g., predicting 19 days when the answer is 18), the model's response is still correct.

Question: {question}

Correct Answer: {answer}

Model Response: {response}

Is the model response correct? Answer yes or no only.""",
        
        'knowledge-update': """I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response contains some previous information along with an updated answer, the response should be considered as correct as long as the updated answer is the required answer.

Question: {question}

Correct Answer: {answer}

Model Response: {response}

Is the model response correct? Answer yes or no only.""",
        
        'single-session-preference': """I will give you a question, a rubric for desired personalized response, and a response from a model. Please answer yes if the response satisfies the desired response. Otherwise, answer no. The model does not need to reflect all the points in the rubric. The response is correct as long as it recalls and utilizes the user's personal information correctly.

Question: {question}

Rubric: {answer}

Model Response: {response}

Is the model response correct? Answer yes or no only.""",
    }
    
    def __init__(self, judge_backend: LLMBackend):
        """
        Initialize with LLM backend for judging.
        
        Args:
            judge_backend: LLM backend to use for evaluation
        """
        self.judge_backend = judge_backend
        self.config: Dict[str, Any] = {}
        self.prompt_templates: Dict[str, str] = {}
    
    def initialize(self, config: Dict[str, Any]) -> None:
        """
        Initialize LLM judge evaluator.
        
        Expected config:
            - prompt_templates: Custom templates for question types (optional)
            - temperature: Temperature for judge LLM (default: 0.0)
            - max_tokens: Max tokens for judge response (default: 10)
        
        Raises:
            RuntimeError: If judge backend is not set
        """
        if self.judge_backend is None:
            raise RuntimeError("Judge backend not set")
        
        self.config = config.copy()
        
        # Use custom templates if provided, otherwise use defaults
        self.prompt_templates = config.get('prompt_templates', self.PROMPT_TEMPLATES.copy())
        
        # Fall back to a generic template for unknown types
        if 'default' not in self.prompt_templates:
            self.prompt_templates['default'] = self.PROMPT_TEMPLATES['single-session-user']
    
    def _get_prompt_template(self, question_type: str) -> str:
        """
        Get the appropriate prompt template for a question type.
        
        Args:
            question_type: Type of question
            
        Returns:
            Prompt template string
        """
        return self.prompt_templates.get(question_type, self.prompt_templates['default'])
    
    def _parse_judge_response(self, response: str) -> bool:
        """
        Parse the judge's yes/no response.
        
        Args:
            response: Raw response from judge LLM
            
        Returns:
            True if answer is correct, False otherwise
        """
        response_lower = response.lower().strip()
        
        # Check for explicit yes/no
        if 'yes' in response_lower:
            return True
        if 'no' in response_lower:
            return False
        
        # If unclear, be conservative and mark as incorrect
        return False
    
    def evaluate_single(
        self,
        question: Question,
        predicted_answer: Answer,
    ) -> Dict[str, Any]:
        """
        Evaluate a single answer using LLM judge.
        
        Args:
            question: Question with ground truth
            predicted_answer: Predicted answer to evaluate
            
        Returns:
            Dict with evaluation results including 'correct' boolean
        """
        # Get the appropriate prompt template
        template = self._get_prompt_template(question.question_type)
        
        # Format the prompt
        prompt = template.format(
            question=question.question_text,
            answer=question.ground_truth_answer,
            response=predicted_answer.answer_text
        )
        
        # Get judgment from LLM
        try:
            judge_response = self.judge_backend.generate(
                prompt,
                temperature=self.config.get('temperature', 0.0),
                max_tokens=self.config.get('max_tokens', 10)
            )
            
            # Parse yes/no
            is_correct = self._parse_judge_response(judge_response)
            
            result = {
                'question_id': question.question_id,
                'question_type': question.question_type,
                'correct': is_correct,
                'judge_response': judge_response,
                'ground_truth': question.ground_truth_answer,
                'predicted': predicted_answer.answer_text,
            }
            
        except Exception as e:
            # If evaluation fails, mark as incorrect and log error
            result = {
                'question_id': question.question_id,
                'question_type': question.question_type,
                'correct': False,
                'error': str(e),
                'ground_truth': question.ground_truth_answer,
                'predicted': predicted_answer.answer_text,
            }
        
        return result
    
    def aggregate_results(
        self,
        results: List[Dict[str, Any]],
        questions: List[Question],
    ) -> EvaluationResult:
        """
        Aggregate individual results into overall metrics.
        
        Args:
            results: List of per-question results
            questions: List of original questions
            
        Returns:
            EvaluationResult with aggregated metrics
        """
        # Overall accuracy
        total_correct = sum(1 for r in results if r.get('correct', False))
        total_questions = len(results)
        overall_accuracy = total_correct / total_questions if total_questions > 0 else 0.0
        
        # Per-type accuracy
        type_stats: Dict[str, Dict[str, Any]] = {}
        for result in results:
            q_type = result.get('question_type', 'unknown')
            
            if q_type not in type_stats:
                type_stats[q_type] = {'correct': 0, 'total': 0}
            
            type_stats[q_type]['total'] += 1
            if result.get('correct', False):
                type_stats[q_type]['correct'] += 1
        
        # Calculate accuracy per type
        per_type_metrics = {}
        for q_type, stats in type_stats.items():
            accuracy = stats['correct'] / stats['total'] if stats['total'] > 0 else 0.0
            per_type_metrics[q_type] = {
                'accuracy': accuracy,
                'correct': stats['correct'],
                'total': stats['total'],
            }
        
        # Create EvaluationResult
        evaluation_result = EvaluationResult(
            overall_metrics={
                'accuracy': overall_accuracy,
                'total_correct': total_correct,
                'total_questions': total_questions,
            },
            per_question_results=results,
            per_type_metrics=per_type_metrics,
            evaluator_name='LLMJudgeEvaluator',
            metadata={
                'judge_backend': self.judge_backend.get_model_info().get('model', 'unknown'),
                'question_types': list(type_stats.keys()),
            }
        )
        
        return evaluation_result
    
    def get_name(self) -> str:
        """Get evaluator name"""
        return "LLMJudgeEvaluator-" + self.judge_backend.get_model_info().get('model', 'unknown')
    
    def supports_batch_evaluation(self) -> bool:
        """LLM judge can support batch evaluation"""
        return False
    
    def evaluate_batch(
        self,
        questions: List[Question],
        predicted_answers: List[Answer],
    ) -> List[Dict[str, Any]]:
        """
        Evaluate a batch of questions (for API efficiency).
        
        Args:
            questions: List of questions
            predicted_answers: List of predicted answers
            
        Returns:
            List of evaluation results
        """
        # TODO: Implement batch evaluation
        # Option: Send multiple requests in parallel
        pass
