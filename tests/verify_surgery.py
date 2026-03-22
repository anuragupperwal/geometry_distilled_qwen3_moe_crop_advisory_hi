import torch
from litgpt.model import GPT
from litgpt.config import Config

def verify():
    ckpt_path = "checkpoints/Qwen/Qwen3-0.6B-moe-initial/lit_model.pth"
    # 1. Physical Check: Does the file exist and have the right keys?
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    has_experts = any("mlp.experts.7" in k for k in sd.keys())
    print(f"Surgery Check: {'✅ Experts found' if has_experts else '❌ Surgery failed'}")

    # 2. Logic Check: Does the model actually run?
    conf = Config.from_name("Qwen3-0.6B-MoE")
    model = GPT(conf)
    model.load_state_dict(sd)
    
    x = torch.randint(0, conf.vocab_size, (1, 8))
    out = model(x)
    print(f"Forward Pass: ✅ Success. Output shape: {out.shape}")

if __name__ == "__main__":
    verify()