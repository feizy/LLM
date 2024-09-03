import argparse
import numpy as np
from transformers import AutoTokenizer


def main(tokenizer_path, file_path, num_tokens):
    # Initialize the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    # Read the binary file
    with open(file_path, 'rb') as f:
        data = np.fromfile(f, dtype=np.uint32)

    # Print the shape of the loaded data
    print(data.shape)

    # Decode and print the specified number of tokens
    print(tokenizer.decode(data[:num_tokens]))


if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Decode tokens from a binary file using a specified tokenizer.")
    parser.add_argument('--tokenizer_path', type=str, default="Qwen/CodeQwen1.5-7B-Chat", help="Path to the pretrained tokenizer.")
    parser.add_argument('--file_path', type=str, required=True, help="Path to the binary file containing tokens.")
    parser.add_argument('--num_tokens', type=int, default=1000, help="Number of tokens to decode and print.")

    # Parse the arguments
    args = parser.parse_args()

    # Call the main function with the parsed arguments
    main(args.tokenizer_path, args.file_path, args.num_tokens)