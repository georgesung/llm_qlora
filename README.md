# Fine-tuning LLMs using QLoRA
## Setup
`pip install -r requirements.txt`

## Run training
Generally, run
```
python train.py <config_file>
```

For exmaple, to fine-tune OpenLLaMA-7B on the wizard_vicuna_70k_unfiltered dataset, run
```
python train.py configs/open_llama_7b_qlora_uncensored.yaml
```
