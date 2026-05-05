# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import json

from transformers import AutoTokenizer

# [Configuration]
# Use a generic text token ID. We avoid 0-10 because they are often reserved
# for special tokens (BOS, EOS, PAD, UNK). 100 is typically a safe, standard subword.
DUMMY_TOKEN_ID = 100


def main():
    parser = argparse.ArgumentParser(
        description="Generate decode-only JSONL dataset for vLLM benchmark")
    parser.add_argument("--model",
                        required=True,
                        type=str,
                        help="Model name or path for Tokenizer")
    parser.add_argument("--input-len",
                        required=True,
                        type=int,
                        help="Exact number of input tokens")
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=1000,
        help="Number of identical prompts to generate",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="/tmp/decode_only_dataset.jsonl",
        help="Path to save the JSONL",
    )
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model,
                                              trust_remote_code=True)

    # Generate exact sequence length
    input_tokens = [DUMMY_TOKEN_ID] * args.input_len
    input_str = tokenizer.decode(input_tokens)

    # vLLM 'custom' dataset expects JSONL format.
    # We provide purely the "prompt" string.
    with open(args.output_file, "w") as f:
        for _ in range(args.num_prompts):
            json_record = {"prompt": input_str}
            f.write(json.dumps(json_record) + "\n")

    print(
        f"[Data Prep] Successfully generated Custom JSONL dataset: EXACTLY {args.input_len} Input Tokens."
    )
    print(f"[Data Prep] Saved to {args.output_file}")


if __name__ == "__main__":
    main()
