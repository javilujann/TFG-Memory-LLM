# Data Directory

This directory contains datasets used for evaluating memory systems.

## Structure

Each dataset should be organized in its own subfolder:

```
data/
├── longmemeval/           # LongMemEval dataset
│   ├── longmemeval_s_cleaned.json
│   ├── longmemeval_m_cleaned.json
│   └── longmemeval_oracle.json
├── other_dataset/         # Add other datasets as needed
│   └── ...
└── README.md             # This file
```

## Adding New Datasets

1. Create a new subfolder with a descriptive name
2. Place your dataset files inside
3. Add a README.md in the subfolder describing the dataset format
4. Implement a corresponding reader in `src/readers/` if needed

## Note

**Dataset contents are not tracked in git** to avoid repository bloat and potential licensing issues. Make sure to:
- Download datasets separately
- Store them locally in the appropriate subfolder
- Never commit actual dataset files
