import json

import torch
from embeddings_classifier.trainer import LinearClassifier
from embeddings_classifier.embeddings_generator import cohere_embeddings

# Load the trained model
model = LinearClassifier(in_features=1024, hidden_dim=512, num_classes=8)
model.load_state_dict(torch.load('embeddings_classifier/best_text_classifier.pth'))
model.eval()  # Set the model to evaluation mode


def classify_text(embeddings):
    """Classify the text using the trained model."""
    with torch.no_grad():
        inputs = torch.tensor(embeddings, dtype=torch.float32).unsqueeze(0)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        return predicted.item()


def evaluate_validation_set():
    """Evaluate the test set using the trained model."""
    from utils import load_json_to_dict
    from tqdm import tqdm
    data = load_json_to_dict('data/embeddings_test.json')
    results = []
    for item in tqdm(data):
        embeddings = item['embeddings']
        sentiment_label = classify_text(torch.tensor(embeddings))
        results.append({'id': item['id'], 'result': sentiment_label})
    with open('embeddings_classifier/linear_results.json', 'w', encoding='utf8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    text = "I hate this product, it's stupid and useless."
    new_embeddings = cohere_embeddings([text])
    class_label = classify_text(torch.tensor(new_embeddings[0]))
    print(f'Predicted class label: {class_label}')

    # evaluate_validation_set()
