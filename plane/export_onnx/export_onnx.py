import os
import argparse
import torch
import torch.nn as nn

from wheel_legged_gym.rsl_rl.modules.actor_critic_sequence import ActorCriticSequence

# =============================
# 配置（与你原来一致）
# =============================
NUM_OBS = 25
OBS_HISTORY_LEN = 5
LATENT_DIM = 3
NUM_ACTIONS = 6

NUM_ENCODER_OBS = NUM_OBS * OBS_HISTORY_LEN
DEFAULT_ONNX_DIR = "./"


# =============================
# ONNX Wrapper（两输入版）
# =============================
class PolicyONNXWrapper(nn.Module):
    def __init__(self, actor_critic):
        super().__init__()
        self.encoder = actor_critic.encoder
        self.actor = actor_critic.actor

    def forward(self, obs, obs_history):
        latent = self.encoder(obs_history)
        actions = self.actor(torch.cat([obs, latent], dim=-1))
        return actions


def export_onnx(pt_path, onnx_path):
    # 1. 构建网络
    model = ActorCriticSequence(
        num_obs=NUM_OBS,
        num_critic_obs=1,
        num_actions=NUM_ACTIONS,
        num_encoder_obs=NUM_ENCODER_OBS,
        latent_dim=LATENT_DIM,
        encoder_hidden_dims=[128, 64],
        actor_hidden_dims=[128, 64, 32],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )

    # 2. 加载 checkpoint
    ckpt = torch.load(pt_path, map_location="cpu")
    state_dict = ckpt["model_state_dict"]

    # 只保留 encoder + actor
    state_dict = {
        k: v for k, v in state_dict.items()
        if not k.startswith("critic.")
    }

    model.load_state_dict(state_dict, strict=False)
    model.eval()

    policy = PolicyONNXWrapper(model).eval()
    onnx_dir = os.path.dirname(onnx_path)
    if onnx_dir:
        os.makedirs(onnx_dir, exist_ok=True)

    # 3. dummy 输入
    dummy_obs = torch.zeros(1, NUM_OBS)
    dummy_obs_history = torch.zeros(1, NUM_ENCODER_OBS)

    # 4. 导出 ONNX
    torch.onnx.export(
        policy,
        (dummy_obs, dummy_obs_history),
        onnx_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["obs", "obs_history"],
        output_names=["actions"],
        dynamic_axes={
            "obs": {0: "batch"},
            "obs_history": {0: "batch"},
            "actions": {0: "batch"},
        },
    )

    print("✅ ONNX exported to:", onnx_path)


# =============================
# 命令行入口
# =============================
if __name__ == "__main__":
    parser = argparse.ArgumentParser("Export Wheel-Legged policy to ONNX")

    parser.add_argument(
        "--load_run",
        type=str,
        required=True,
        help="Run index under logs/wheel_legged_vmc_flat (e.g. 4)",
    )
    parser.add_argument(
        "--checkpoint",
        type=int,
        required=True,
        help="Checkpoint index (e.g. 600 -> model_600.pt)",
    )
    parser.add_argument(
        "--log_root",
        type=str,
        default="logs/wheel_legged",
        help="Root log directory",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="policy.onnx",
        help="Output ONNX path",
    )

    args = parser.parse_args()

    # 拼 checkpoint 路径
    pt_path = os.path.join(
        args.log_root,
        str(args.load_run),
        f"model_{args.checkpoint}.pt",
    )

    if not os.path.isfile(pt_path):
        raise FileNotFoundError(f"Checkpoint not found: {pt_path}")

    onnx_path = args.out
    if not os.path.isabs(onnx_path):
        onnx_path = os.path.join(DEFAULT_ONNX_DIR, onnx_path)

    export_onnx(
        pt_path=pt_path,
        onnx_path=onnx_path,
    )
