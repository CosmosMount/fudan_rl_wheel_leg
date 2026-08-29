"""Minimal Sequence-PPO implementation and legacy-compatible ONNX export."""

from __future__ import annotations

import argparse
import copy
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from mjlab.rl import MjlabOnPolicyRunner, RslRlBaseRunnerCfg
from rsl_rl.storage import RolloutStorage
from tensordict import TensorDict
from torch import nn
from torch.distributions import Normal


def _mlp(dims: tuple[int, ...]) -> nn.Sequential:
  layers: list[nn.Module] = []
  for in_dim, out_dim in zip(dims[:-2], dims[1:-1], strict=True):
    layers.extend((nn.Linear(in_dim, out_dim), nn.ELU()))
  layers.append(nn.Linear(dims[-2], dims[-1]))
  return nn.Sequential(*layers)


class SequencePolicy(nn.Module):
  """History encoder plus actor with the original two-input deployment contract."""

  def __init__(self) -> None:
    super().__init__()
    Normal.set_default_validate_args(False)
    self.encoder = _mlp((125, 128, 64, 3))
    self.actor = _mlp((28, 128, 64, 32, 6))
    self.std = nn.Parameter(torch.full((6,), 0.5))

  def mean(self, obs: torch.Tensor, history: torch.Tensor, detach: bool = True) -> torch.Tensor:
    latent = self.encoder(history)
    if detach:
      latent = latent.detach()
    return self.actor(torch.cat((obs, latent), dim=-1))

  def forward(self, observations: TensorDict | dict[str, torch.Tensor]) -> torch.Tensor:
    return self.mean(observations["policy"], observations["history"])

  @property
  def output_std(self) -> torch.Tensor:
    return self.std.detach()

  def as_onnx(self, verbose: bool = False) -> OnnxPolicy:
    del verbose
    return OnnxPolicy(self.encoder, self.actor)


class SequenceCritic(nn.Module):
  def __init__(self) -> None:
    super().__init__()
    self.critic = _mlp((144, 256, 128, 64, 1))

  def forward(self, critic_obs: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
    return self.critic(torch.cat((critic_obs, latent.detach()), dim=-1))


class OnnxPolicy(nn.Module):
  input_names = ["obs", "obs_history"]
  output_names = ["actions"]

  def __init__(self, encoder: nn.Module, actor: nn.Module) -> None:
    super().__init__()
    self.encoder = encoder
    self.actor = actor

  def forward(self, obs: torch.Tensor, obs_history: torch.Tensor) -> torch.Tensor:
    return self.actor(torch.cat((obs, self.encoder(obs_history)), dim=-1))

  def get_dummy_inputs(self) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.zeros(1, 25), torch.zeros(1, 125)


@dataclass
class SequencePpoCfg:
  class_name: str = "wbr_mjlab.rl:SequencePPO"
  # Large rollouts already contain 393k plane transitions. Three passes lower
  # the update-to-data ratio while retaining four independently sized batches.
  num_learning_epochs: int = 3
  num_mini_batches: int = 4
  clip_param: float = 0.2
  gamma: float = 0.99
  lam: float = 0.95
  value_loss_coef: float = 1.0
  entropy_coef: float = 0.01
  learning_rate: float = 1e-3
  encoder_learning_rate: float = 1e-3
  max_grad_norm: float = 1.0
  encoder_max_grad_norm: float = 0.1
  use_clipped_value_loss: bool = True
  schedule: str = "adaptive"
  desired_kl: float = 0.005
  rnd_cfg: dict | None = None
  share_cnn_encoders: bool = False


@dataclass
class SequenceRunnerCfg(RslRlBaseRunnerCfg):
  class_name: str = "wbr_mjlab.rl:SequenceRunner"
  seed: int = 1
  num_steps_per_env: int = 48
  max_iterations: int = 50_000
  save_interval: int = 100
  experiment_name: str = "wbr_sequence_ppo"
  logger: str = "tensorboard"
  upload_model: bool = False
  clip_actions: float | None = 100.0
  obs_groups: dict[str, tuple[str, ...]] = field(
    default_factory=lambda: {
      "actor": ("policy", "history"),
      "critic": ("critic", "history"),
    }
  )
  algorithm: SequencePpoCfg = field(default_factory=SequencePpoCfg)


def sequence_runner_cfg(mode: str) -> SequenceRunnerCfg:
  cfg = SequenceRunnerCfg()
  cfg.experiment_name = f"wbr_{mode}"
  return cfg


class SequencePPO:
  """PPO whose encoder is frozen during PPO and supervised afterward."""

  def __init__(
    self,
    policy: SequencePolicy,
    critic: SequenceCritic,
    storage: RolloutStorage,
    *,
    device: str = "cpu",
    multi_gpu_cfg: dict | None = None,
    **algorithm_cfg: Any,
  ) -> None:
    self.cfg = SequencePpoCfg(**algorithm_cfg)
    self.device = device
    self.policy = policy.to(device)
    self.actor = self.policy
    self.critic = critic.to(device)
    self.storage = storage
    self.transition = RolloutStorage.Transition()
    self._ppo_parameters = [
      *self.policy.actor.parameters(),
      self.policy.std,
      *self.critic.parameters(),
    ]
    self.optimizer = torch.optim.Adam(self._ppo_parameters, lr=self.cfg.learning_rate)
    self.encoder_optimizer = torch.optim.Adam(
      self.policy.encoder.parameters(), lr=self.cfg.encoder_learning_rate
    )
    self.learning_rate = self.cfg.learning_rate
    self.rnd = None
    self.intrinsic_rewards = None
    self.is_multi_gpu = multi_gpu_cfg is not None
    self.gpu_global_rank = 0 if multi_gpu_cfg is None else multi_gpu_cfg["global_rank"]
    self.gpu_world_size = 1 if multi_gpu_cfg is None else multi_gpu_cfg["world_size"]

  @staticmethod
  def construct_algorithm(obs: TensorDict, env: Any, cfg: dict, device: str) -> SequencePPO:
    policy, critic = SequencePolicy(), SequenceCritic()
    storage = RolloutStorage(
      "rl", env.num_envs, cfg["num_steps_per_env"], obs, [env.num_actions], device
    )
    return SequencePPO(
      policy,
      critic,
      storage,
      device=device,
      multi_gpu_cfg=cfg.get("multi_gpu"),
      **cfg["algorithm"],
    )

  def _distribution(self, obs: TensorDict) -> Normal:
    with torch.no_grad():
      latent = self.policy.encoder(obs["history"])
    mean = self.policy.actor(torch.cat((obs["policy"], latent), dim=-1))
    return Normal(mean, self.policy.std.expand_as(mean))

  def _value(self, obs: TensorDict) -> torch.Tensor:
    with torch.no_grad():
      latent = self.policy.encoder(obs["history"])
    return self.critic(obs["critic"], latent)

  def _actor_critic(self, obs: TensorDict) -> tuple[Normal, torch.Tensor]:
    """Evaluate the shared frozen history latent once for actor and critic."""
    with torch.no_grad():
      latent = self.policy.encoder(obs["history"])
    mean = self.policy.actor(torch.cat((obs["policy"], latent), dim=-1))
    dist = Normal(mean, self.policy.std.expand_as(mean))
    return dist, self.critic(obs["critic"], latent)

  def act(self, obs: TensorDict) -> torch.Tensor:
    dist, values = self._actor_critic(obs)
    actions = dist.sample()
    self.transition.actions = actions.detach()
    self.transition.values = values.detach()
    self.transition.actions_log_prob = dist.log_prob(actions).sum(-1).detach()
    self.transition.distribution_params = (dist.mean.detach(), dist.stddev.detach())
    self.transition.observations = obs
    return actions

  def process_env_step(
    self,
    obs: TensorDict,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    extras: dict[str, torch.Tensor],
  ) -> None:
    del obs
    self.transition.rewards = rewards.clone()
    self.transition.dones = dones
    if "time_outs" in extras:
      self.transition.rewards += self.cfg.gamma * torch.squeeze(
        self.transition.values * extras["time_outs"].unsqueeze(1).to(self.device), 1
      )
    self.storage.add_transition(self.transition)
    self.transition.clear()

  def compute_returns(self, obs: TensorDict) -> None:
    st = self.storage
    last_values = self._value(obs).detach()
    advantage = torch.zeros_like(last_values)
    for step in reversed(range(st.num_transitions_per_env)):
      next_values = last_values if step == st.num_transitions_per_env - 1 else st.values[step + 1]
      alive = 1.0 - st.dones[step].float()
      delta = st.rewards[step] + alive * self.cfg.gamma * next_values - st.values[step]
      advantage = delta + alive * self.cfg.gamma * self.cfg.lam * advantage
      st.returns[step] = advantage + st.values[step]
    st.advantages = st.returns - st.values
    st.advantages = (st.advantages - st.advantages.mean()) / (st.advantages.std() + 1e-8)

  def _adapt_learning_rate(
    self, old_mean: torch.Tensor, old_std: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
  ) -> None:
    if self.cfg.schedule != "adaptive" or self.cfg.desired_kl is None:
      return
    with torch.no_grad():
      kl = (
        torch.log(std / old_std + 1e-5)
        + (old_std.square() + (old_mean - mean).square()) / (2.0 * std.square())
        - 0.5
      )
      kl_mean = kl.sum(-1).mean()
      if kl_mean > 2.0 * self.cfg.desired_kl:
        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
      elif 0.0 < kl_mean < 0.5 * self.cfg.desired_kl:
        self.learning_rate = min(1e-2, self.learning_rate * 1.5)
      for group in self.optimizer.param_groups:
        group["lr"] = self.learning_rate

  def ppo_update(self) -> dict[str, float]:
    totals = {"value": 0.0, "surrogate": 0.0, "entropy": 0.0}
    generator = self.storage.mini_batch_generator(
      self.cfg.num_mini_batches, self.cfg.num_learning_epochs
    )
    for batch in generator:
      assert batch.observations is not None
      dist, values = self._actor_critic(batch.observations)
      log_prob = dist.log_prob(batch.actions).sum(-1)
      old_mean, old_std = batch.old_distribution_params
      self._adapt_learning_rate(old_mean, old_std, dist.mean, dist.stddev)
      ratio = torch.exp(log_prob - batch.old_actions_log_prob.squeeze(-1))
      advantage = batch.advantages.squeeze(-1)
      surrogate = -advantage * ratio
      clipped = -advantage * ratio.clamp(1.0 - self.cfg.clip_param, 1.0 + self.cfg.clip_param)
      surrogate_loss = torch.maximum(surrogate, clipped).mean()
      if self.cfg.use_clipped_value_loss:
        value_clipped = batch.values + (values - batch.values).clamp(
          -self.cfg.clip_param, self.cfg.clip_param
        )
        value_loss = torch.maximum(
          (values - batch.returns).square(), (value_clipped - batch.returns).square()
        ).mean()
      else:
        value_loss = (values - batch.returns).square().mean()
      entropy = dist.entropy().sum(-1).mean()
      loss = (
        surrogate_loss + self.cfg.value_loss_coef * value_loss - self.cfg.entropy_coef * entropy
      )
      self.optimizer.zero_grad()
      loss.backward()
      self.reduce_parameters()
      nn.utils.clip_grad_norm_(
        self._ppo_parameters,
        self.cfg.max_grad_norm,
      )
      self.optimizer.step()
      totals["value"] += value_loss.item()
      totals["surrogate"] += surrogate_loss.item()
      totals["entropy"] += entropy.item()
    count = self.cfg.num_learning_epochs * self.cfg.num_mini_batches
    return {key: value / count for key, value in totals.items()}

  def auxiliary_update(self) -> float:
    total = 0.0
    generator = self.storage.mini_batch_generator(
      self.cfg.num_mini_batches, self.cfg.num_learning_epochs
    )
    for batch in generator:
      assert batch.observations is not None
      prediction = self.policy.encoder(batch.observations["history"])
      target = batch.observations["critic"][:, :3].detach()
      loss = torch.nn.functional.mse_loss(prediction, target)
      self.encoder_optimizer.zero_grad()
      loss.backward()
      self.reduce_parameters()
      nn.utils.clip_grad_norm_(self.policy.encoder.parameters(), self.cfg.encoder_max_grad_norm)
      self.encoder_optimizer.step()
      total += loss.item()
    return total / (self.cfg.num_learning_epochs * self.cfg.num_mini_batches)

  def update(self) -> dict[str, float]:
    losses = self.ppo_update()
    losses["encoder"] = self.auxiliary_update()
    self.storage.clear()
    return losses

  def train_mode(self) -> None:
    self.policy.train()
    self.critic.train()

  def eval_mode(self) -> None:
    self.policy.eval()
    self.critic.eval()

  def get_policy(self) -> SequencePolicy:
    return self.policy

  def save(self) -> dict[str, Any]:
    state = {
      "format_version": 1,
      "policy_state_dict": self.policy.state_dict(),
      "critic_state_dict": self.critic.state_dict(),
      "optimizer_state_dict": self.optimizer.state_dict(),
      "encoder_optimizer_state_dict": self.encoder_optimizer.state_dict(),
      "normalization_state_dict": {},
      "torch_rng_state": torch.get_rng_state(),
      "numpy_rng_state": np.random.get_state(),
      "python_rng_state": random.getstate(),
    }
    if torch.cuda.is_available():
      state["cuda_rng_state"] = torch.cuda.get_rng_state_all()
    return state

  def load(self, state: dict, load_cfg: dict | None = None, strict: bool = True) -> bool:
    cfg = load_cfg
    if cfg is None:
      cfg = {
        "actor": True,
        "critic": True,
        "optimizer": True,
        "iteration": True,
        "rng": True,
      }
    load_model = cfg.get("model", False)
    if cfg.get("actor", load_model):
      self.policy.load_state_dict(state["policy_state_dict"], strict=strict)
    if cfg.get("critic", load_model):
      self.critic.load_state_dict(state["critic_state_dict"], strict=strict)
    if cfg.get("optimizer", False):
      self.optimizer.load_state_dict(state["optimizer_state_dict"])
      self.encoder_optimizer.load_state_dict(state["encoder_optimizer_state_dict"])
      self.learning_rate = self.optimizer.param_groups[0]["lr"]
    if cfg.get("rng", False):
      torch.set_rng_state(state["torch_rng_state"])
      np.random.set_state(state["numpy_rng_state"])
      random.setstate(state["python_rng_state"])
      if "cuda_rng_state" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda_rng_state"])
    return cfg.get("iteration", False)

  def compile(self, mode: str | None = None) -> None:
    del mode

  def broadcast_parameters(self) -> None:
    if self.is_multi_gpu:
      params = [self.policy.state_dict(), self.critic.state_dict()]
      torch.distributed.broadcast_object_list(params, src=0)
      self.policy.load_state_dict(params[0])
      self.critic.load_state_dict(params[1])

  def reduce_parameters(self) -> None:
    if not self.is_multi_gpu:
      return
    for param in [*self.policy.parameters(), *self.critic.parameters()]:
      if param.grad is not None:
        torch.distributed.all_reduce(param.grad)
        param.grad /= self.gpu_world_size


class SequenceRunner(MjlabOnPolicyRunner):
  """mjlab runner with dynamic-batch two-input ONNX export."""

  def add_git_repo_to_log(self, repo_file_path: str) -> None:
    # RSL-RL decodes the complete working-tree diff with surrogate escapes and
    # later writes it as strict UTF-8. A migration that deletes binary legacy
    # assets can therefore abort training before iteration zero. Checkpoints and
    # structured configs remain logged; skip only this optional source snapshot.
    self.logger.git_status_repos.clear()
    del repo_file_path

  def export_policy_to_onnx(
    self, path: str, filename: str = "policy.onnx", verbose: bool = False
  ) -> None:
    output = Path(path) / filename
    output.parent.mkdir(parents=True, exist_ok=True)
    mode = self.env.unwrapped.cfg.actions["hybrid"].mode
    export_onnx(self.alg.get_policy(), output, verbose=verbose, mode=mode)


def export_onnx(
  policy: SequencePolicy, output: Path, verbose: bool = False, *, mode: str | None = None
) -> None:
  model = copy.deepcopy(policy.as_onnx()).cpu().eval()
  torch.onnx.export(
    model,
    model.get_dummy_inputs(),
    str(output),
    export_params=True,
    opset_version=18,
    verbose=verbose,
    input_names=model.input_names,
    output_names=model.output_names,
    dynamic_axes={"obs": {0: "batch"}, "obs_history": {0: "batch"}, "actions": {0: "batch"}},
    dynamo=False,
  )
  if mode is not None:
    import json

    import onnx

    from .sim2sim import METADATA_KEY, policy_contract

    exported = onnx.load(str(output))
    onnx.helper.set_model_props(exported, {METADATA_KEY: json.dumps(policy_contract(mode))})
    onnx.checker.check_model(exported)
    onnx.save(exported, str(output))


def export_main() -> None:
  parser = argparse.ArgumentParser(description="Export a WBR Sequence-PPO checkpoint")
  parser.add_argument(
    "--task", required=True, choices=("Mjlab-Velocity-Flat-WBR", "Mjlab-Jump-Flat-WBR")
  )
  parser.add_argument("--checkpoint", required=True, type=Path)
  parser.add_argument("--output", required=True, type=Path)
  parser.add_argument("--verify", action="store_true", help="Compare ONNX and Torch on fixed random inputs")
  args = parser.parse_args()
  mode = "plane" if args.task == "Mjlab-Velocity-Flat-WBR" else "jump"
  checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
  policy = SequencePolicy()
  policy.load_state_dict(checkpoint["policy_state_dict"])
  args.output.parent.mkdir(parents=True, exist_ok=True)
  export_onnx(policy, args.output, mode=mode)
  print(f"Exported {mode} policy: {args.output}")
  if args.verify:
    import numpy as np

    from .sim2sim import OnnxPolicy

    deployed = OnnxPolicy(args.output, mode)
    generator = torch.Generator().manual_seed(17)
    model = policy.as_onnx().eval()
    max_error = 0.0
    with torch.inference_mode():
      for _ in range(64):
        obs = torch.randn(1, 25, generator=generator)
        history = torch.randn(1, 125, generator=generator)
        expected = model(obs, history)[0].clamp(-100, 100).numpy()
        actual = deployed(obs.numpy(), history.numpy())
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
        max_error = max(max_error, float(np.abs(actual - expected).max()))
    print(f"ONNX verification passed (64 inputs, max absolute error {max_error:.3g})")


__all__ = [
  "SequencePPO",
  "SequencePolicy",
  "SequenceRunner",
  "export_onnx",
  "sequence_runner_cfg",
]


if __name__ == "__main__":
  export_main()
