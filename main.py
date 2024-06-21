import os
import sys
import argparse

from embeddings_classifier.inference import classify_text
from embeddings_classifier.embeddings_generator import cohere_embeddings


def validate_file(filepath):
    # Check file type
    if not filepath.lower().endswith('.txt'):
        raise ValueError("File must be a .txt file")

    # Check file existence
    if not os.path.isfile(filepath):
        raise FileNotFoundError("File does not exist")

    # Check file size (limit to 5MB)
    if os.path.getsize(filepath) > 5 * 1024 * 1024:
        raise ValueError("File is too large (limit: 5MB)")

    # Check path length (limit to 512 characters)
    if len(filepath) > 512:
        raise ValueError("File path is too long (limit: 255 characters)")


def main():
    parser = argparse.ArgumentParser(description="Classify text from a .txt file.")
    parser.add_argument('--file', type=str, required=True, help="Path to the .txt file")

    args = parser.parse_args()
    filepath = args.file

    try:
        validate_file(filepath)
    except (ValueError, FileNotFoundError) as e:
        print(f"Error: {e}")
        sys.exit(1)

    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            text = file.read()
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)

    try:
        embeddings = cohere_embeddings([text])
        classification = classify_text(embeddings[0])
    except Exception as e:
        print(f"Error classifying text: {e}")
        sys.exit(1)

    print(f"The classified label for the document is: {classification}")


if __name__ == "__main__":
    main()
