import json

import cohere
from configuration import config
from utils import load_json_to_dict

co = cohere.Client(config['embeddings_classifier']['COHERE_API_KEY'])


def cohere_embeddings(passages, input_type="classification"):
    """Compute embeddings using the COHERE API."""
    embeds = co.embed(texts=passages, model=config['embeddings_classifier']['cohere_embeddings_model'],
                      input_type=input_type).embeddings
    return embeds


def add_embeddings_to_dict(data_dicts, embeddings_list):
    """Add embeddings to the data json."""
    for i, data_dict in enumerate(data_dicts):
        data_dict['embeddings'] = embeddings_list[i]
    return data_dicts


if __name__ == "__main__":
    json_data_path = config['general']['path_to_training_data']
    passages_dict = load_json_to_dict(json_data_path)
    passages = [item['text'] for item in passages_dict]
    # embeddings = cohere_embeddings(passages)
    # passages_dict = add_embeddings_to_dict(passages_dict, embeddings)
    # with open('data/embeddings_training.json', 'w', encoding='utf-8') as file:
    #     json.dump(passages_dict, file, ensure_ascii=False)
