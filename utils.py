import json

import yaml

from sklearn.metrics import f1_score, precision_recall_fscore_support

CLASSES_DICT = {
    0: '0',
    1: '1',
    2: '2',
    3: '3',
    4: '4',
    5: '5',
    6: '6',
    7: '7'
}


def calculate_f1_scores(y_pred, y_true):
    # Calculate micro F1 score
    micro_f1 = f1_score(y_true, y_pred, average='micro')

    # Calculate macro F1 score
    macro_f1 = f1_score(y_true, y_pred, average='macro')

    # Calculate precision, recall, and F1 score for each class
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=list(CLASSES_DICT.keys()))

    # Convert the precision, recall, and F1 scores to dictionaries
    class_f1_scores = {CLASSES_DICT[label]: {'precision': precision[i], 'recall': recall[i], 'f1': f1[i]} for i, label
                       in enumerate(CLASSES_DICT.keys())}

    return micro_f1, macro_f1, class_f1_scores


def calculate_f1_scores_from_json(predictions_json_path, ground_truth_json_path):
    with open(predictions_json_path, 'r', encoding='utf8') as f:
        predictions = json.load(f)
    with open(ground_truth_json_path, 'r', encoding='utf8') as f:
        ground_truth = json.load(f)

    y_pred = [entry['result'] for entry in predictions]
    y_true = [entry['result'] - 1 for entry in ground_truth]
    _micro_f1, _macro_f1, _class_f1_scores = calculate_f1_scores(y_pred, y_true)
    return _micro_f1, _macro_f1, _class_f1_scores


def load_yaml_to_dict(yaml_file):
    with open(yaml_file, 'r', encoding='utf-8') as f:
        data = yaml.load(f, Loader=yaml.FullLoader)

    return data


def load_json_to_dict(json_file):
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


if __name__ == '__main__':
    predictions_json_path = 'embeddings_classifier/linear_results.json'
    ground_truth_json_path = 'data/embeddings_test.json'

    micro_f1, macro_f1, class_f1_scores = calculate_f1_scores_from_json(predictions_json_path, ground_truth_json_path)

    print("Micro F1 Score:", micro_f1)
    print("Macro F1 Score:", macro_f1)
    print("\nClass-wise F1 Scores:")
    for class_name, scores in class_f1_scores.items():
        print(f"{class_name}: Precision={scores['precision']}, Recall={scores['recall']}, F1={scores['f1']}")
