# devquasar.com
import torch
import yaml
from transformers import (AutoTokenizer, LlamaForCausalLM, BitsAndBytesConfig, pipeline)
import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")

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
    raw_output = pipe(chat_history + '\n' + get_prompt(prompt), stop_sequence=PROMPT_STOP)


    return raw_output

if __name__ == "__main__":
    q_config = BitsAndBytesConfig(load_in_8bit=True)

    print("Load model")
    model = "DevQuasar/llama3_8b_chat_brainstorm"
    tokenizer = AutoTokenizer.from_pretrained(model)
    model = LlamaForCausalLM.from_pretrained(model, device_map="auto", quantization_config=q_config)

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
    try:
        chat_history = ''
        while True:
            user_input = input("[USER] (Ctrl+C to exit):\n")
            chat_response = get_llm_response(user_input, chat_history, debug=debug)
            chat_history = chat_response[-1]['generated_text']
            if debug:
                print(f'[USER]: {user_input}\n[CHAT]: {chat_response}\n')
            else:
                last_entry = chat_response[-1]
                generated_text = last_entry['generated_text']
                last_response_index = generated_text.rfind('### RESPONSE:')
                if last_response_index != -1:
                    last_response = generated_text[last_response_index + len('### RESPONSE:'):].strip()
                    print(f'\n[CHAT]: {last_response}\n')
    except KeyboardInterrupt:
        print("\nExiting...")

