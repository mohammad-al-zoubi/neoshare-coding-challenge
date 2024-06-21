# TODO:
#  1. Convert data to json, add ids & split data
#  2. Generate embeddings
#  3. Train embeddings classifier
#  4. Evaluate embeddings classifier
#  5. Write inference code
#  6. Prepare presentation
import json
import random

from pathlib import Path


def convert_data_to_json(txt_file_path):
    """Convert the training text file to a json file."""
    json_data = []
    with open(txt_file_path, 'r') as file:
        lines = file.readlines()
        for i, line in enumerate(lines[1:]):
            data_point = {'id': i, 'result': int(line.strip()[:2].strip()), 'text': line.strip()[2:]}
            json_data.append(data_point)
    save_path = Path(txt_file_path).parent / 'file.json'
    with open(save_path, 'w', encoding='utf-8') as file:
        json.dump(json_data, file, ensure_ascii=False, indent=4)


def split_data(file_path, test_proportion):
    """Split the data into training and test sets."""
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    # Shuffle the data to ensure random distribution
    random.shuffle(data)

    # Calculate the split index
    split_index = int(len(data) * (1 - test_proportion))

    # Split the data into training and test sets
    training_data = data[:split_index]
    test_data = data[split_index:]

    training_save_path = Path(file_path).parent / 'training_data.json'
    test_save_path = Path(file_path).parent / 'test_data.json'
    with open(training_save_path, 'w', encoding='utf-8') as train_file:
        json.dump(training_data, train_file, indent=4)

    with open(test_save_path, 'w', encoding='utf-8') as test_file:
        json.dump(test_data, test_file, indent=4)


if __name__ == '__main__':
    path = Path(r'data/file.json')
    # convert_data_to_json(path)
    # split_data(path, 0.1)
