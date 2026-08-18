import os
import torch
import torch.nn as nn

LOG_DIR = "/root/gpufree-data/wheel_leg/wheel_leg/Wheel-Legged-Gym-master/logs/wheel_legged_vmc_flat/flat_locomotion_jump"
CKPT_PATH = os.path.join(LOG_DIR, "model_600.pt")

OUT_DIR = "/root/gpufree-data/wheel_leg/wheel_leg/Wheel-Legged-Gym-master/logs/wheel_legged_vmc_flat/exported/policies"
OUT_PATH = os.path.join(OUT_DIR, "encoder_1.pt")

device = "cpu"

ckpt = torch.load(CKPT_PATH, map_location=device)
sd = ckpt["model_state_dict"]

# 你的 config：obs_history_length=5, num_observations=27 => 135
# latent_dim=3, encoder_hidden_dims=[128,64]
# 所以 encoder 结构：135 -> 128 -> 64 -> 3
encoder = nn.Sequential(
    nn.Linear(135, 128),
    nn.ELU(),
    nn.Linear(128, 64),
    nn.ELU(),
    nn.Linear(64, 3),
).to(device).eval()

# 从 checkpoint 填权重：你 ckpt 里就是 encoder.0 / encoder.2 / encoder.4
with torch.no_grad():
    encoder[0].weight.copy_(sd["encoder.0.weight"])
    encoder[0].bias.copy_(sd["encoder.0.bias"])
    encoder[2].weight.copy_(sd["encoder.2.weight"])
    encoder[2].bias.copy_(sd["encoder.2.bias"])
    encoder[4].weight.copy_(sd["encoder.4.weight"])
    encoder[4].bias.copy_(sd["encoder.4.bias"])

# 导出 TorchScript
os.makedirs(OUT_DIR, exist_ok=True)
dummy = torch.zeros(1, 135, dtype=torch.float32)
jit_enc = torch.jit.trace(encoder, dummy)
jit_enc.save(OUT_PATH)

print("Saved:", OUT_PATH)
print("schema:", jit_enc.forward.schema)
