import torch
from isaacgym.gymapi import (
    KEY_F,
    KEY_P,
    KEY_L,
    KEY_J,
    KEY_R,
    KEY_U,
    KEY_W,
    KEY_S,
    KEY_A,
    KEY_D,
    KEY_UP,
    KEY_DOWN,
    KEY_X,
    KEY_H,
)


def _clip_(x, lo, hi):
    return torch.clamp(x, lo, hi)


class KeyboardCtrl:
    """
    commands 映射（只用前三个 range）：
      commands[:, 0] -> lin_vel_x
      commands[:, 1] -> ang_vel_yaw
      commands[:, 2] -> height
    """

    def __init__(self, env, env_cfg, **kwargs):
        self.env = env
        self.env_cfg = env_cfg
        self.num_actions = int(kwargs["num_actions"])
        self.agent_model = kwargs.get("agent_model", None)
        self.FPV = bool(kwargs.get("FPV", False))
        self.record = False

        # 你只用到前三个 range
        self.lin_lo, self.lin_hi = env_cfg.commands.ranges.lin_vel_x
        self.yaw_lo, self.yaw_hi = env_cfg.commands.ranges.ang_vel_yaw
        self.h_lo, self.h_hi = env_cfg.commands.ranges.height

        # 默认高度设置为中值（可按需改）
        if self.env.commands.shape[1] > 2:
            mid_h = 0.5 * (self.h_lo + self.h_hi)
            self.env.commands[:, 2] = mid_h

        key_actions = {
            KEY_P: "push_robot",
            KEY_L: "press_robot",
            KEY_J: "action_jitter",
            KEY_R: "agent_full_reset",
            KEY_U: "full_reset",
            KEY_W: "forward",
            KEY_S: "backward",
            KEY_A: "leftturn",
            KEY_D: "rightturn",
            KEY_UP: "height_up",
            KEY_DOWN: "height_down",
            KEY_X: "stop",
            KEY_F: "FPV",
            KEY_H: "record",
        }

        for key, action in key_actions.items():
            env.gym.subscribe_viewer_keyboard_event(env.viewer, key, action)

    def _clamp_commands(self):
        # lin_vel_x
        self.env.commands[:, 0] = _clip_(self.env.commands[:, 0], self.lin_lo, self.lin_hi)
        # ang_vel_yaw
        self.env.commands[:, 1] = _clip_(self.env.commands[:, 1], self.yaw_lo, self.yaw_hi)
        # height（如果有这个通道）
        if self.env.commands.shape[1] > 2:
            self.env.commands[:, 2] = _clip_(self.env.commands[:, 2], self.h_lo, self.h_hi)

    def run(self):
        for ui_event in self.env.gym.query_viewer_action_events(self.env.viewer):
            if ui_event.value == 0:
                continue

            a = ui_event.action

            if a == "push_robot":
                if hasattr(self.env, "_push_robots"):
                    self.env._push_robots()

            elif a == "action_jitter":
                # 正确的 action shape: (num_envs, num_actions)
                rand_action = torch.randn(
                    (self.env.num_envs, self.num_actions),
                    device=self.env.device,
                )
                _ = self.env.step(rand_action)

            elif a == "agent_full_reset":
                if self.agent_model is not None and hasattr(self.agent_model, "reset"):
                    self.agent_model.reset()

            elif a == "full_reset":
                if self.agent_model is not None and hasattr(self.agent_model, "reset"):
                    self.agent_model.reset()
                _ = self.env.reset()

            elif a == "stop":
                # 停止只影响前三个通道：x/yaw/height
                self.env.commands[:, 0] = 0.0
                self.env.commands[:, 1] = 0.0
                if self.env.commands.shape[1] > 2:
                    # 高度回到中值更合理（不然 0 可能超 range）
                    mid_h = 0.5 * (self.h_lo + self.h_hi)
                    self.env.commands[:, 2] = mid_h

            elif a == "forward":
                self.env.commands[:, 0] += 0.2

            elif a == "backward":
                self.env.commands[:, 0] -= 0.2

            elif a == "leftturn":
                self.env.commands[:, 1] += 0.5

            elif a == "rightturn":
                self.env.commands[:, 1] -= 0.5

            elif a == "height_up":
                if self.env.commands.shape[1] > 2:
                    self.env.commands[:, 2] += 0.005

            elif a == "height_down":
                if self.env.commands.shape[1] > 2:
                    self.env.commands[:, 2] -= 0.005

            elif a == "FPV":
                self.FPV = not self.FPV

            elif a == "record":
                self.record = not self.record

            # 最后统一 clamp（只用前三个 range）
            self._clamp_commands()
