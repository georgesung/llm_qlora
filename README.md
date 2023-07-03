# Fine-tuning LLMs using QLoRA
## Setup
`pip install -r requirements.txt`

## Run training
```
python train.py <config_file>
```

For exmaple, to fine-tune OpenLLaMA-7B on the wizard_vicuna_70k_unfiltered dataset, run
```
python train.py configs/open_llama_7b_qlora_uncensored.yaml
```

## Push model to HuggingFace Hub
Follow instructions [here](https://huggingface.co/docs/hub/repositories-getting-started#terminal).

## Example inference results
See this [Colab notebook](https://colab.research.google.com/drive/1IlpeofYD9EU6dNHyKKObZhIzkBMyqlUS?usp=sharing).
