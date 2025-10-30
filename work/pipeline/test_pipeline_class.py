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
    """Run the complete pipeline using EvaluationPipeline class"""
    print("🚀 Running EvaluationPipeline")
    
    # Configuration
    dataset_path = "./data/longmemeval/longmemeval_oracle.json"

    # Step 1: Create Pipeline
    print("=" * 70)
    print("🚀 Creating Pipeline")
    print("=" * 70)

    pipeline_config = load_config("./config/First_config.yaml") # Load pipeline configuration
    print("✅ Configuration loaded")

    pipeline = create_pipeline_from_config(pipeline_config) # Create pipeline instance from config
    print("✅ Pipeline created")

    # Step 2: Run the pipeline
    try:
        results = pipeline.run(dataset_path)

    except Exception as e:
        print(f"\n❌ Pipeline execution failed: {e}")
        import traceback
        traceback.print_exc()

    # Step 3: Print summary of results
    print("\n🚀 Pipeline Results Summary")
    for eval_name, result in results.items():
        print(result.summary())

if __name__ == "__main__":
    main()
