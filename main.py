import os
import sys
import logging
import argparse

from embeddings_classifier.inference import classify_text
from embeddings_classifier.embeddings_generator import cohere_embeddings

log = logging.getLogger(__name__)


def validate_file(filepath):

    # Check file type
    if not filepath.lower().endswith('.txt'):
        log.error("File must be a .txt file")
        raise ValueError("File must be a .txt file")

    # Check file existence
    if not os.path.isfile(filepath):
        log.error("File does not exist")
        raise FileNotFoundError("File does not exist")

    # Check file size (limit to 5MB)
    if os.path.getsize(filepath) > 5 * 1024 * 1024:
        log.error("File is too large (limit: 5MB)")
        raise ValueError("File is too large (limit: 5MB)")

    # Check path length (limit to 512 characters)
    if len(filepath) > 512:
        log.error("File path is too long (limit: 512 characters)")
        raise ValueError("File path is too long (limit: 512 characters)")


def main():
    parser = argparse.ArgumentParser(description="Classify text from a .txt file.")
    parser.add_argument('--file', type=str, required=True, help="Path to the .txt file")

    args = parser.parse_args()
    filepath = args.file

    try:
        log.info(f"Validating file: {filepath}")
        validate_file(filepath)
    except (ValueError, FileNotFoundError) as e:
        log.error(f"Error: {e}")
        sys.exit(1)

    try:
        log.info(f"Reading file: {filepath}")
        with open(filepath, 'r', encoding='utf-8') as file:
            text = file.read()
    except Exception as e:
        log.error(f"Error reading file: {e}")
        sys.exit(1)

    try:
        log.info("Classifying text")
        embeddings = cohere_embeddings([text])
        classification = classify_text(embeddings[0])
    except Exception as e:
        log.error(f"Error classifying text: {e}")
        sys.exit(1)

    log.info(f"The classified label for the document is: {classification}")
    print(f"The classified label for the document is: {classification}")


if __name__ == "__main__":
    main()
