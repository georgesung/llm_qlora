import torch
import transformers
from datasets import load_dataset
from datasets.dataset_dict import DatasetDict
from peft import (LoraConfig, PeftModel, get_peft_model,
                  prepare_model_for_kbit_training)
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, LlamaForCausalLM, LlamaTokenizer)


class QloraTrainer:
    def __init__(self, config: dict):
        self.config = config
        self.tokenizer = None
        self.base_model = None
        self.adapter_model = None
        self.merged_model = None

    def load_base_model(self):
        model_id = self.config["base_model"]

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        if "model_family" in self.config and self.config["model_family"] == "llama":
            tokenizer = LlamaTokenizer.from_pretrained(model_id)
            model = LlamaForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map={"":0})
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map={"":0})

        if not tokenizer.pad_token:
            # Add padding token if missing, e.g. for llama tokenizer
            #tokenizer.pad_token = tokenizer.eos_token  # https://github.com/huggingface/transformers/issues/22794
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        model.gradient_checkpointing_enable()
        model = prepare_model_for_kbit_training(model)

        self.tokenizer = tokenizer
        self.base_model = model

    def load_adapter_model(self, adapter_path: str):
        """ Load pre-trained lora adapter """
        self.adapter_model = PeftModel.from_pretrained(self.base_model, adapter_path)

    def train(self):
        # Set up lora config or load pre-trained adapter
        if self.adapter_model is None:
            config = LoraConfig(
                r=8,
                lora_alpha=32,
                target_modules=self.config["target_modules"],
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM"
            )
            model = get_peft_model(self.base_model, config)
        else:
            model = self.adapter_model
        self._print_trainable_parameters(model)

        print("Start data preprocessing")
        # TODO: Expand this to cover more dataset types and processing patterns
        data = self._process_vicuna_data()

        print("Start training")
        trainer = transformers.Trainer(
            model=model,
            train_dataset=data["train"],
            args=transformers.TrainingArguments(
                per_device_train_batch_size=1,
                gradient_accumulation_steps=4,
                warmup_steps=100,
                #max_steps=200,  # short run for debugging
                num_train_epochs=1,  # full run
                learning_rate=2e-4,
                fp16=True,
                logging_steps=20,
                output_dir=self.config["trainer_output_dir"],
                report_to="tensorboard",
                #optim="adamw"
            ),
            data_collator=transformers.DataCollatorForLanguageModeling(self.tokenizer, mlm=False),
        )
        model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
        trainer.train()

        model_save_path = f"{self.config['model_output_dir']}/{self.config['model_name']}_adapter"
        trainer.save_model(model_save_path)
        self.adapter_model = model
        print(f"Training complete, adapter model saved in {model_save_path}")

    def merge_and_save(self):
        """ Merge base model and adapter, save to disk """
        # Cannot merge when base model loaded in 8-bit/4-bit mode, so load separately
        model_id = self.config["base_model"]
        if "model_family" in self.config and self.config["model_family"] == "llama":
            base_model = LlamaForCausalLM.from_pretrained(model_id, device_map="cpu")
        else:
            base_model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cpu")

        adapter_save_path = f"{self.config['model_output_dir']}/{self.config['model_name']}_adapter"
        model = PeftModel.from_pretrained(base_model, adapter_save_path)

        self.merged_model = model.merge_and_unload()  # note it's on CPU, don't run inference on it

        model_save_path = f"{self.config['model_output_dir']}/{self.config['model_name']}"
        self.merged_model.save_pretrained(model_save_path)
        self.tokenizer.save_pretrained(model_save_path)

    def push_to_hub(self):
        """ Push merged model to HuggingFace Hub """
        raise NotImplementedError("push_to_hub not implemented yet")

    def _print_trainable_parameters(self, model):
        """
        Prints the number of trainable parameters in the model.
        """
        trainable_params = 0
        all_param = 0
        for _, param in model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(
            f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
        )

    def _generate_prompt(self, convo: list, eos_token: str, instruct: bool = False) -> str:
        convo_text = ""
        for turn in convo:
            entity = turn["from"]
            value = turn["value"]

            if entity == "human":
                convo_text += "### HUMAN:\n"
                end_token = ""
            elif entity == "gpt":
                convo_text += "### RESPONSE:\n"
                end_token = eos_token  # LLM should stop its output after the response
            else:
                print(f"WARNING: uknown entity {entity}")
                convo_text += f"### {entity.upper()}:\n"
                end_token = ""

            convo_text += value + end_token + "\n\n"

            if instruct and entity == "gpt":
                return convo_text
        return convo_text

    def _process_vicuna_data(self) -> DatasetDict:
        if "model_context_window" in self.config:
            context_window = self.config["model_context_window"]
        else:
            context_window = self.tokenizer.model_max_length

        data = load_dataset(self.config["dataset"])
        data = data.map(lambda data_point: self.tokenizer(
            self._generate_prompt(
                data_point["conversations"],
                self.tokenizer.eos_token, 
                instruct=self.config["instruct"]),
            max_length=context_window,
            truncation=True,
        ))
        return data
