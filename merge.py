import argparse
import yaml
from QloraTrainer import QloraTrainer
import os
import warnings
warnings.filterwarnings("ignore")


def read_yaml_file(file_path):
    with open(file_path, 'r') as file:
        try:
            data = yaml.safe_load(file)
            return data
        except yaml.YAMLError as e:
            print(f"Error reading YAML file: {e}")


if __name__ == "__main__":
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", help="Path to the config YAML file")
    args = parser.parse_args()

    config = read_yaml_file(args.config_path)
    trainer = QloraTrainer(config)

    print("Load base model")
    trainer.load_base_model()

    print("Start merging")
    trainer.merge_and_save()
