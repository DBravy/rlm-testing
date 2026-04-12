# Project: rlm

## Environment

- **Python environment**: Conda environment named `rlm`
- **Python binary**: `/opt/anaconda3/envs/rlm/bin/python`
- **Activate**: `conda activate rlm`

When running scripts or checking packages, use the conda env python directly if conda activate isn't available in the shell:
```
/opt/anaconda3/envs/rlm/bin/python script.py
/opt/anaconda3/envs/rlm/bin/pip show <package>
```

## Key Dependencies

- `trl` v1.1.0 (GRPOTrainer, GRPOConfig)
- `datasets` (Hugging Face)
- `transformers` (Hugging Face)

## Notes

- `GRPOConfig` in trl v1.1.0 does **not** have `max_prompt_length`. Use `max_completion_length` only.
