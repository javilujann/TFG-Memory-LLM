#!/usr/bin/env python3
"""
Full Pipeline Test using EvaluationPipeline Class

Complete end-to-end test using the pipeline orchestrator.
"""

import sys
from pathlib import Path

# Add the work directory to path
work_dir = Path(__file__).parent.parent
sys.path.insert(0, str(work_dir))

from src.utils import load_config, create_pipeline_from_config


def main():
    """Test the complete pipeline using EvaluationPipeline class"""
    print("🚀 Testing EvaluationPipeline Class")
    print("=" * 70)
    
    # Configuration
    dataset_path = "./data/longmemeval/longmemeval_oracle.json"
    pipeline_config = load_config("./config/First_config.yaml")

    print(f"✅ Pipeline config created: {pipeline_config.experiment_name}")
    
    # Step 3: Create and Run Pipeline
    print("\n" + "=" * 70)
    print("🚀 Creating and Running Pipeline")
    print("=" * 70)

    pipeline = create_pipeline_from_config(pipeline_config)

    print("✅ Pipeline created")
    
    # Run the pipeline
    try:
        results = pipeline.run(dataset_path)
        
        # Display results
        print("\n" + "=" * 70)
        print("📈 RESULTS SUMMARY")
        print("=" * 70)
        
        for eval_name, eval_result in results.items():
            print(f"\n{eval_name}:")
            print(f"{'='*70}")
            print(f"Overall Accuracy: {eval_result.overall_metrics['accuracy']:.2%}")
            print(f"Correct: {eval_result.overall_metrics['total_correct']}/{eval_result.overall_metrics['total_questions']}")
            
            print(f"\nPer-Type Accuracy:")
            for q_type, metrics in eval_result.per_type_metrics.items():
                acc = metrics['accuracy']
                correct = metrics['correct']
                total = metrics['total']
                print(f"  {q_type}: {acc:.2%} ({correct}/{total})")
            
            print(f"\nMetadata:")
            for key, value in eval_result.metadata.items():
                print(f"  {key}: {value}")
        
        print("\n" + "=" * 70)
        print("✅ PIPELINE TEST COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        
        print(f"\n📁 Results saved to: {pipeline.output_dir}")
        print(f"   - Answers: {pipeline_config.experiment_name}_answers.jsonl")
        print(f"   - Evaluations: {pipeline_config.experiment_name}_*_results.json")
        print(f"   - Final: {pipeline_config.experiment_name}_final_results.json")
        
    except Exception as e:
        print(f"\n❌ Pipeline execution failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
