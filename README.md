# Fine-tuning LLMs using QLoRA
## Setup
First, make sure you are using python 3.8+. If you're using python 3.7, see the Troubleshooting section below.

`pip install -r requirements.txt`

## Run training
```
python train.py <config_file>
```

For exmaple, to fine-tune Llama3-8B on the wizard_vicuna_70k_unfiltered dataset, run
```
python train.py configs/llama3_8b_chat_uncensored.yaml
```

## Push model to HuggingFace Hub
Follow instructions [here](https://huggingface.co/docs/hub/repositories-getting-started#terminal).

## Models trained on HuggingFace Hub
| Model name | Config file | URL |
|----------|----------|----------|
| llama3_8b_chat_uncensored | configs/llama3_8b_chat_uncensored.yaml | https://huggingface.co/georgesung/llama3_8b_chat_uncensored |
| llama2_7b_openorca_35k | configs/llama2_7b_openorca_35k.yaml | https://huggingface.co/georgesung/llama2_7b_openorca_35k |
| llama2_7b_chat_uncensored | configs/llama2_7b_chat_uncensored.yaml | https://huggingface.co/georgesung/llama2_7b_chat_uncensored |
| open_llama_7b_qlora_uncensored | configs/open_llama_7b_qlora_uncensored.yaml | https://huggingface.co/georgesung/llama2_7b_openorca_35k |


## Inference
Simple sanity check:
```
python inference.py
```

For notebooks with example inference results, see `inference.ipynb` and this [Colab notebook](https://colab.research.google.com/drive/1IlpeofYD9EU6dNHyKKObZhIzkBMyqlUS?usp=sharing).

## Blog post
Blog post describing the process of QLoRA fine tuning: https://georgesung.github.io/ai/qlora-ift/

## Converting to GGUF and quantizing the model
Download and build [llama.cpp](https://github.com/ggerganov/llama.cpp), and follow the instructions on their README to convert the model to GGUF and quantize to desired specs.

*Tip*: If llama.cpp gives an error saying the number of tokens is different between the model and tokenizer.json, it could be because we added a pad token (e.g. for training Llama). One work-around is to copy the original tokenizer.json from the base model (you can find the base model in huggingface cache at `~/.cache/huggingface/`) to the new model's location, but make sure to back-up your tokenizer.json!

*Tip*: Llama3 uses BPE tokenizer, make sure to specify `--vocab-type bpe` when converting to GGUF

## Troubleshooting
### Issues with python 3.7
If you're using python 3.7, you will install `transformers 4.30.x`, since `transformers >=4.31.0` [no longer supports python 3.7](https://github.com/huggingface/transformers/releases/tag/v4.31.0). If you then install the latest version of `peft`, the GPU memory consumption will be higher than usual. The work-around is to use an older version of `peft` to go along with the older `transformers` version you installed. Update your `requirements.txt` as follows:
```
transformers==4.30.2
git+https://github.com/huggingface/peft.git@86290e9660d24ef0d0cedcf57710da249dd1f2f4
```
Of course, make sure to remove the original lines with `transformers` and `peft`, and run `pip install -r requirements.txt`
