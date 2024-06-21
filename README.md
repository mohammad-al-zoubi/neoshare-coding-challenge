# Text Classification Coding Challenge - Neoshare AG

To classify a .txt file:

`python main.py --file "document.txt"`

Output:

```
The classified label for the document is: 2
```


To build the docker image:

`docker build -t text_classification:latest .`


To start the text classification server:

`docker run -p 8000:8000 text_classification`


To call the service:

```python3
from server.client import call_predict_endpoint

text = "I feel like dancing in the rain."
mode = "linear"
class_label = call_predict_endpoint(text, mode)
print(f"Class: {class_label}")
```

Output:

```
Class: 7
```

Note: In this solution the embeddings which are used by the linear classifier are generated via the API. Hence, for the linear classifier prediction a valid API key for Cohere is required. These need to be set in config.yaml.