# Inference Configuration Summary

## Datasets
- psytar: max_length=195, batch_size=16, temperature=0.9
- Daniel-ML: max_length=135, batch_size=16, temperature=0.9
- asylax: max_length=4141, batch_size=4, temperature=0.9
- n2c2_2008: max_length=3072, batch_size=4, temperature=0.9

## Models
- Llama-3.2-1B-Instruct: batch_multiplier=1.0
- Llama-3.2-3B-Instruct: batch_multiplier=0.5
- Llama-3.1-8B-Instruct: batch_multiplier=0.25
- Llama-3.3-70B-Instruct: batch_multiplier=0.125

## Epsilon Values
- 0.5, 1, 2, 4, nodp

## Expected Model Paths
Models should be located at:
`research/synthetic-text-generation-with-DP/result/{dataset}/{model}/{epsilon}/final/`

## Generated Scripts
Total scripts generated: 16

## Usage
```bash
# Run all combinations
./run_all_inference.sh

# Run specific dataset
./run_all_inference.sh psytar

# Run specific dataset-model combination
./run_all_inference.sh psytar Llama-3.2-1B-Instruct

# Run individual script
./generate_psytar_Llama-3.2-1B-Instruct.sh
```
