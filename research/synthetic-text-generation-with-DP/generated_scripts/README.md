# Training Configuration Summary

## Datasets
- psytar: max_length=195, base_batch_size=16
- Daniel-ML: max_length=135, base_batch_size=16
- asylax: max_length=4141, base_batch_size=4
- n2c2_2008: max_length=3072, base_batch_size=4

## Models
- Llama-3.2-1B-Instruct: batch_multiplier=1.0, grad_accum=1
- Llama-3.2-3B-Instruct: batch_multiplier=0.5, grad_accum=2
- Llama-3.1-8B-Instruct: batch_multiplier=0.25, grad_accum=4
- Llama-3.3-70B-Instruct: batch_multiplier=0.125, grad_accum=8

## Generated Scripts
Total production scripts: 16
Total debug scripts: 16
Total scripts: 32

## Debug vs Production
**Debug scripts:**
- Train for only 0.01 epochs (~few steps)
- Test only epsilon=4 and nodp
- Save to `debug_result/` folder
- Auto-delete models after successful completion
- Keep failed models for debugging
- Perfect for validating setup

**Production scripts:**
- Train for full 10 epochs
- Test all epsilon values: 0.5, 1, 2, 4, nodp
- Save to `result/` folder
- Keep all models
- For actual experiments

## Usage
```bash
# Test everything quickly (recommended first step)
./run_debug_training.sh

# Test specific dataset
./run_debug_training.sh psytar

# Test specific combination
./run_debug_training.sh psytar Llama-3.2-1B-Instruct

# Run production training (after debug passes)
./run_all_training.sh

# Run production with debug flag
./run_all_training.sh '' '' debug

# Run individual debug script
./debug_run_psytar_Llama-3.2-1B-Instruct.sh
```
