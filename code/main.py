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


def main(config_path: str):
    """
    Run the complete pipeline using EvaluationPipeline class
    
    Args:
        config_path: Path to the configuration YAML file
    """
    print("🚀 Running EvaluationPipeline\n")
    
    # Step 1: Create Pipeline
    print("=" * 70)
    print("🚀 Creating Pipeline")
    print("=" * 70)
    print(f"📄 Loading configuration from: {config_path}")

    pipeline_config = load_config(config_path) # Load pipeline configuration
    print("✅ Configuration loaded")

    pipeline = create_pipeline_from_config(pipeline_config) # Create pipeline instance from config
    print("✅ Pipeline created")

    # Step 2: Run the pipeline
    try:
        results = pipeline.run()

    except Exception as e:
        print(f"\n❌ Pipeline execution failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # Step 3: Print summary of results
    print("\n🚀 Pipeline Results Summary")
    for eval_name, result in results.items():
        print(result.summary())


if __name__ == "__main__":
    # Check if config file path is provided as command line argument
    if len(sys.argv) < 2:
        print("❌ Error: Configuration file path is required")
        print("Usage: python main.py <config_file_path>")
        print("Example: python main.py ./config/First.yaml")
        sys.exit(1)
    
    config_file = sys.argv[1]
    
    # Check if config file exists
    config_path = Path(config_file)
    if not config_path.exists():
        print(f"❌ Error: Configuration file not found: {config_file}")
        sys.exit(1)
      
    # Run the pipeline with the provided config
    main(config_file)
