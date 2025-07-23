# Training Configuration Summary

## Datasets
- psytar: max_length=195, base_batch_size=32
- Daniel-ml: max_length=135, base_batch_size=32
- asylax: max_length=4141, base_batch_size=2
- n2c2: max_length=3072, base_batch_size=2

## Models
- Llama-3.2-1B-Instruct: batch_multiplier=1.0, grad_accum=1
- Llama-3.2-3B-Instruct: batch_multiplier=0.5, grad_accum=2
- Llama-3.1-8B-Instruct: batch_multiplier=0.25, grad_accum=4
- Llama-3.3-70B-Instruct: batch_multiplier=0.125, grad_accum=8

## Generated Scripts
Total scripts generated: 16

## Usage
```bash
# Run all combinations
./run_all_training.sh

# Run specific dataset
./run_all_training.sh psytar

# Run specific dataset-model combination
./run_all_training.sh psytar Llama-3.2-1B-Instruct

# Run individual script
./run_psytar_Llama-3.2-1B-Instruct.sh
```
