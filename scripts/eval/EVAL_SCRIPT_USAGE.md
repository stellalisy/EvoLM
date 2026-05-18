# Evaluation Script Usage Guide

## eval_parallel_by_task_by_step.sh

### Description
Submit parallel evaluation jobs - one per (checkpoint, task) combination.
Now supports evaluating a specific step only!

### Syntax
```bash
./eval_parallel_by_task_by_step.sh <exp_name> <checkpoint_dir> <output_dir> [step_number] [tasks...]
```

### Parameters
- `exp_name`: Experiment name (used for job naming)
- `checkpoint_dir`: Directory containing step_* subdirectories
- `output_dir`: Output directory for evaluation results
- `step_number` (optional): Specific step to evaluate (e.g., 1000, 500)
- `tasks...` (optional): Specific tasks to evaluate (default: all tulu_3_dev tasks)

### Examples

#### 1. Evaluate ALL steps with ALL tasks
```bash
./eval_parallel_by_task_by_step.sh main_alt50 \
    /checkpoint/.../checkpoints \
    /checkpoint/.../olmes_eval
```
This submits: `num_steps × num_tasks` jobs

#### 2. Evaluate ONLY step 1000 with ALL tasks
```bash
./eval_parallel_by_task_by_step.sh main_alt50 \
    /checkpoint/.../checkpoints \
    /checkpoint/.../olmes_eval \
    1000
```
This submits: `1 × 11 = 11` jobs (one per task)

#### 3. Evaluate ONLY step 500 with specific tasks
```bash
./eval_parallel_by_task_by_step.sh seq_iq_s2 \
    /checkpoint/.../checkpoints \
    /checkpoint/.../olmes_eval \
    500 \
    gsm8k::tulu \
    minerva_math::tulu \
    codex_humaneval::tulu \
    codex_humanevalplus::tulu
```
This submits: `1 × 4 = 4` jobs (only specified tasks)

#### 4. Evaluate ALL steps with specific tasks
```bash
./eval_parallel_by_task_by_step.sh main_alt50 \
    /checkpoint/.../checkpoints \
    /checkpoint/.../olmes_eval \
    drop::llama3 \
    gsm8k::tulu
```
This submits: `num_steps × 2` jobs

### Default Tasks (if not specified)
- `gsm8k::tulu`
- `drop::llama3`
- `minerva_math::tulu`
- `codex_humaneval::tulu`
- `codex_humanevalplus::tulu`
- `ifeval::tulu`
- `popqa::tulu`
- `mmlu:mc::tulu`
- `alpaca_eval_v2::tulu`
- `bbh:cot-v1::tulu`
- `truthfulqa::tulu`

### Use Cases

#### For Sequential Experiments (evaluate step 500 only)
```bash
./eval_parallel_by_task_by_step.sh seq_iq_s2 \
    /checkpoint/.../seq_iq_checkpoints \
    /checkpoint/.../seq_iq_olmes_eval \
    500
```

#### For Main Experiments (evaluate step 1000 only)
```bash
./eval_parallel_by_task_by_step.sh main_alt50 \
    /checkpoint/.../main_checkpoints \
    /checkpoint/.../main_olmes_eval \
    1000
```

#### To add missing GSM8K, MATH, Code to sequential at step 500
```bash
./eval_parallel_by_task_by_step.sh seq_iq_s2 \
    /checkpoint/.../seq_iq_checkpoints \
    /checkpoint/.../seq_iq_olmes_eval \
    500 \
    gsm8k::tulu \
    minerva_math::tulu \
    codex_humaneval::tulu \
    codex_humanevalplus::tulu
```

### Features
- ✅ Smart skip logic: Won't re-evaluate if task already has complete metrics
- ✅ Task-aware expected counts (minerva_math expects 7 files, mmlu expects 57, etc.)
- ✅ Safe job names (special characters replaced)
- ✅ Confirmation prompt before submitting

### Monitoring
```bash
# Monitor all eval jobs for an experiment
squeue -u $USER | grep ev_main_alt50

# Check specific step
squeue -u $USER | grep ev_main_alt50_s1000

# View logs
ls -lt log/eval/ev_main_alt50*.{out,err}
```
