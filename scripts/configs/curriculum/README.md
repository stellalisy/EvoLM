# Curriculum Scheduling

This directory contains configuration files for curriculum-based training schedules.

## 1. Judge Size Curriculum (IMPLEMENTED)

Start with a large judge model (e.g., 32B) and progressively switch to smaller judges (14B → 8B → 4B).

**Status**: ✅ IMPLEMENTED

**Config Variables**:
```bash
# Comma-separated list of model names (largest to smallest)
JUDGE_SIZE_CURRICULUM="Qwen/Qwen3-32B,Qwen/Qwen3-14B,Qwen/Qwen3-8B,Qwen/Qwen3-4B"

# Optional: Explicit cycle indices for switches (must start at 0, strictly increasing)
# If not set, switches are evenly distributed across total cycles
JUDGE_CURRICULUM_SCHEDULE="0,10,20,30"  # cycles at which to switch
```

**Usage**:
```bash
./scripts/launch.sh \
    configs/alternating_training.sh \
    configs/curriculum/judge_size_curriculum.sh
```

**How It Works**:
1. Training starts with the first (largest) model in the curriculum
2. At each cycle boundary, checks if a model switch is needed based on the schedule
3. When switching:
   - Pauses both policy and rubric training
   - Shuts down current judge engines
   - Creates new engines with the smaller model
   - Updates both actors with new judge references
   - Resumes training
4. Curriculum state is logged to wandb (`curriculum/current_model_idx`, `curriculum/current_model`)

**Notes**:
- `RUBRIC_JUDGE_MODEL` should be set to the first model in the curriculum
- Model switches have ~5 second pause for GPU memory cleanup
- All other judge engine settings (num_engines, tensor_parallel_size, etc.) are inherited

---

## 2. External Judge Mixing (NOT YET IMPLEMENTED)

Blend a strong external judge (GPT-4) with the learned rubric judge, decaying the external judge weight over training.

**Status**: ❌ NOT IMPLEMENTED

Mix a strong external judge (GPT-4) with the learned rubric judge:
```
J_eff = α_t · J_external + (1 - α_t) · J_rubric
```
where α_t decays from 1.0 to 0.0 over training.

**Required Changes**:
1. Add `external_judge_weight` parameter to reward computation
2. Modify `rubric_judge_rewards.py` to support weighted combination
3. Add decay schedule (linear, exponential, or step-based)

**Config Variables Needed**:
```bash
EXTERNAL_JUDGE_MODEL="gpt-4o"
EXTERNAL_JUDGE_INITIAL_WEIGHT=0.8
EXTERNAL_JUDGE_DECAY="linear"  # or "exponential", "step"
EXTERNAL_JUDGE_DECAY_STEPS=1000
```

**Files to Modify**:
- `scripts/train_rubric_policy_joint.py` - main training loop
- `scripts/base_config.sh` - add new config variables
- `open_instruct/grpo_fast.py` - reward computation

## Implementation Notes

For external judge mixing, consider:
1. **API costs**: GPT-4 calls add significant cost
2. **Latency**: External API calls are slower than local inference
3. **Training stability**: Need smooth decay to avoid abrupt signal changes
