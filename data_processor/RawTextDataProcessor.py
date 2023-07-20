from datasets import Dataset, DatasetDict, load_dataset

from data_processor.DataProcessor import DataProcessor


class RawTextDataProcessor(DataProcessor):
    def __init__(self, config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer

    def get_data(self) -> DatasetDict:
        if "model_context_window" in self.config:
            context_window = self.config["model_context_window"]
        else:
            context_window = self.tokenizer.model_max_length

        # Read each text file and chunk it
        texts = []
        for text_file in self.config["data"]["text_files"]:
            with open(text_file, "r") as file:
                all_text = file.read()

            # Chunk the text
            chunk_char_len = self.config["data"]["chunk_char_len"]
            chunk_char_overlap = self.config["data"]["chunk_char_overlap"]

            i = 0
            while i < len(all_text):
                chunk = ""
                if "chunk_prefix" in self.config["data"]:
                    chunk += self.config["data"]["chunk_prefix"]

                i = max(i - chunk_char_overlap, 0)
                chunk += all_text[i:i+chunk_char_len]
                texts.append(chunk)
                i += chunk_char_len

        # Create HF DatasetsDict
        text_dict = {"text": texts}
        dataset = Dataset.from_dict(text_dict)
        data = DatasetDict({"train": dataset})

        # Tokenize & trucate text to create final DatasetDict
        data = data.map(lambda data_point: self.tokenizer(
            data_point["text"],
            max_length=context_window,
            truncation=True,
        ))
        return data
