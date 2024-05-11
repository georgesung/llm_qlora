import argparse

import torch
import yaml
from transformers import (AutoConfig, AutoModel, AutoModelForCausalLM,
                          AutoTokenizer, GenerationConfig, LlamaForCausalLM,
                          LlamaTokenizer, BitsAndBytesConfig, pipeline)
import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")

"""
Ad-hoc sanity check to see if model outputs something coherent
Not a robust inference platform!
"""

PROMPT_STOP = ["### Assistant:", "### Human:"]

def read_yaml_file(file_path):
    with open(file_path, 'r') as file:
        try:
            data = yaml.safe_load(file)
            return data
        except yaml.YAMLError as e:
            print(f"Error reading YAML file: {e}")

def get_prompt(human_prompt):
    prompt_template=f"### HUMAN:\n{human_prompt}\n\n### RESPONSE:\n"
    return prompt_template

def get_llm_response(prompt, chat_history, debug=False):
    if debug:
        print(f"debug:  {chat_history}  {get_prompt(prompt)}")
    raw_output = pipe(chat_history + '\n' + get_prompt(prompt), stop_sequence=PROMPT_STOP, pad_token_id=14711)


    return raw_output

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", help="Path to the config YAML file")
    args = parser.parse_args()

    config = read_yaml_file(args.config_path)
    q_config = BitsAndBytesConfig(load_in_8bit=True)

    print("Load model")
    model_path = f"{config['model_output_dir']}/{config['model_name']}"
    if "model_family" in config and config["model_family"] == "llama":
        tokenizer = LlamaTokenizer.from_pretrained(model_path)
        model = LlamaForCausalLM.from_pretrained(model_path, device_map="auto", quantization_config=q_config)
        # tokenizer = LlamaTokenizer.from_pretrained('georgesung/llama3_8b_chat_uncensored')
        # model = LlamaForCausalLM.from_pretrained('georgesung/llama3_8b_chat_uncensored', device_map="auto", quantization_config=q_config)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", quantization_config=q_config)
        # tokenizer = AutoTokenizer.from_pretrained('georgesung/llama3_8b_chat_uncensored')
        # model = AutoModelForCausalLM.from_pretrained('georgesung/llama3_8b_chat_uncensored', device_map="auto", quantization_config=q_config)

    pipe = pipeline(
        "text-generation",
        model=model, 
        tokenizer=tokenizer, 
        max_length=512,
        temperature=0.7,
        top_p=0.95,
        repetition_penalty=1.15
    )

    debug = False
    # print(len(tokenizer))  # 32001
    # print(model.config.vocab_size)  # 32000
    # print(tokenizer.get_added_vocab())  # {'<pad>': 32000}
    try:
        chat_history = ''
        while True:
            user_input = input("[USER] (Ctrl+C to exit):\n")
            chat_response = get_llm_response(user_input, chat_history, debug=debug)
            chat_history = chat_response[-1]['generated_text']
            if debug:
                print(f'[USER]: {user_input}\n[CHAT]: {chat_response}\n')
            else:
                # Get the last element of the array
                last_entry = chat_response[-1]

                # Extract the 'generated_text' value
                generated_text = last_entry['generated_text']

                # Find the last occurrence of '### RESPONSE:'
                last_response_index = generated_text.rfind('### RESPONSE:')

                # Extract the text after the last '### RESPONSE:'
                if last_response_index != -1:
                    last_response = generated_text[last_response_index + len('### RESPONSE:'):].strip()
                    print(f'\n[CHAT]: {last_response}\n')
    except KeyboardInterrupt:
        print("\nExiting...")
    # print(get_llm_response("What is your favorite movie?"))
