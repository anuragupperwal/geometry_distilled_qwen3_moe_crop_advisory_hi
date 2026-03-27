import torch
from pathlib import Path
from agri_data import AgriDataset
from litgpt.tokenizer import Tokenizer

DATA_PATH = "data/train_bilingual_mixed_83k_agri65k.parquet"
NUM_SAMPLES = 10

OUTPUT_FILE = Path("tests/debug_dataset_output.txt")

tokenizer = Tokenizer("checkpoints/Qwen/Qwen3-0.6B")
dataset = AgriDataset(DATA_PATH, tokenizer)

with open(OUTPUT_FILE, "w") as f:

    for sample_id in range(NUM_SAMPLES):

        tokens, mask = dataset[sample_id]

        decoded = tokenizer.decode(tokens)

        masked_tokens = tokens[mask == 0]
        loss_tokens = tokens[mask == 1]

        prompt_text = tokenizer.decode(masked_tokens)
        assistant_text = tokenizer.decode(loss_tokens)

        print("prompt_text\n", prompt_text)
        print("assistant_text\n", assistant_text)

        f.write("\n" + "="*80 + "\n")
        f.write(f"SAMPLE {sample_id}\n")
        f.write("="*80 + "\n\n")

        f.write("----- PROMPT (MASKED FROM LOSS) -----\n")
        f.write(prompt_text + "\n\n")

        f.write("----- ASSISTANT TRAINING TARGET (LOSS) -----\n")
        f.write(assistant_text + "\n\n")

        # Try splitting think/advisory for readability
        if "<think>" in assistant_text:

            think_start = assistant_text.find("<think>")
            think_end = assistant_text.find("</think>")

            if think_end != -1:

                thinking = assistant_text[think_start:think_end+8]
                advisory = assistant_text[think_end+8:]

                f.write("----- THINKING -----\n")
                f.write(thinking + "\n\n")

                f.write("----- ADVISORY -----\n")
                f.write(advisory + "\n\n")

        f.write("\n\n")

print(f"\nDataset debug written to: {OUTPUT_FILE}")