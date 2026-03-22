import torch
import torch.nn.functional as F

from litgpt.model import GPT
from litgpt.config import Config
from litgpt.tokenizer import Tokenizer


DEVICE = "cuda"

MODEL_PATH = "checkpoints/Qwen/Qwen3-0.6B-Agri-Distilled/02_03_run_test_80k_E9E703_good_kept_almostthere/step-4200.pth"
TOKENIZER_PATH = "checkpoints/Qwen/Qwen3-0.6B"


print("Loading tokenizer...")
tokenizer = Tokenizer(TOKENIZER_PATH)

print("Loading model...")
config = Config.from_name("Qwen3-0.6B-MoE")
model = GPT(config).to(DEVICE, dtype=torch.bfloat16)

checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
# model.load_state_dict(checkpoint["model_state_dict"], strict=False)
model.load_state_dict(checkpoint, strict=False)
model.eval()

print("\nRunning sanity test...\n")

# Extremely simple prompt
prompt = "Farmer problem: rice leaves turning yellow. Solution:"

input_ids = tokenizer.encode(prompt, bos=False, eos=False).to(DEVICE)
input_ids = input_ids.unsqueeze(0)

print("Prompt:")
print(prompt)
print("\nGenerated:")

for _ in range(50):

    with torch.no_grad():
        logits = model(input_ids)

    next_token_logits = logits[:, -1, :]
    next_token = torch.argmax(next_token_logits, dim=-1)
    print(torch.topk(next_token_logits, 10))
    input_ids = torch.cat([input_ids, next_token.unsqueeze(1)], dim=1)

    token_text = tokenizer.decode(next_token)

    print(token_text, end="", flush=True)

print("\n")