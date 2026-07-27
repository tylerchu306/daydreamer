from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

import embodied


class NexArm(embodied.Env):
    """Adapter connecting NexArmEnv to the DayDreamer Embodied API."""

    def __init__(
        self,
        mode: str = "state",
        size: tuple[int, int] = (64, 64),
        seed: int | None = 0,
        max_episode_steps: int = 300,
        randomize_object: bool = True,
    ) -> None:
        if mode not in ("state", "vision"):
            raise ValueError(
                f"mode must be 'state' or 'vision', got {mode!r}"
            )

        repo = Path(
            os.environ.get(
                "NEXARM_REPO",
                "/kaggle/working/arm-manipulation",
            )
        ).expanduser().resolve()

        if not repo.exists():
            raise FileNotFoundError(
                f"NexArm repository not found: {repo}. "
                "Set the NEXARM_REPO environment variable."
            )

        repo_string = str(repo)
        if repo_string not in sys.path:
            sys.path.insert(0, repo_string)

        # Import only after the NexArm repository is added to sys.path.
        from envs.nexarm_env import NexArmEnv

        scene_path = repo / "assets" / "robot" / "scene.xml"
        if not scene_path.exists():
            raise FileNotFoundError(scene_path)

        self._mode = mode
        self._size = tuple(size)
        self._next_seed = seed
        self._done = True
        self._last_info: dict[str, Any] = {}

        self._joint_delta = 0.04
        self._gripper_threshold = 0.25

        self._previous_distance: float | None = None
        self._previous_object_z: float | None = None
        self._previous_grasped = False

        self._env = NexArmEnv(
            scene_path=scene_path,
            image_height=self._size[0],
            image_width=self._size[1],
            frame_skip=10,
            max_episode_steps=max_episode_steps,
            randomize_object=randomize_object,
        )

    @property
    def obs_space(self) -> dict[str, embodied.Space]:
        # State mode contains privileged simulator information:
        # robot state (6), object xyz (3), grasp-site xyz (3), distance (1).
        vector_size = 13 if self._mode == "state" else 6

        spaces = {
            "vector": embodied.Space(
                np.float32,
                (vector_size,),
            ),
            "reward": embodied.Space(np.float32),
            "is_first": embodied.Space(bool),
            "is_last": embodied.Space(bool),
            "is_terminal": embodied.Space(bool),
        }

        if self._mode == "vision":
            spaces["image"] = embodied.Space(
                np.uint8,
                self._size + (3,),
            )

        return spaces

    @property
    def act_space(self) -> dict[str, embodied.Space]:
        return {
            "action": embodied.Space(
                np.float32,
                (6,),
                np.full((6,), -1.0, np.float32),
                np.full((6,), 1.0, np.float32),
            ),
            "reset": embodied.Space(bool),
        }

    @property
    def info(self) -> dict[str, Any]:
        return self._last_info

    def step(self, action: dict[str, Any]) -> dict[str, Any]:
        if bool(action["reset"]) or self._done:
            raw, info = self._env.reset(seed=self._next_seed)

            # Seed only the first reset. Later resets continue the environment
            # random-number sequence instead of producing the same cube pose.
            self._next_seed = None
            self._done = False
            self._last_info = info

            self._previous_distance = float(
                info["object_to_grasp_distance"]
            )
            self._previous_object_z = float(
                info["object_z"]
            )
            self._previous_grasped = bool(
                getattr(self._env, "_grasped", False)
            )

            return self._convert(
                raw=raw,
                info=info,
                reward=0.0,
                is_first=True,
                is_last=False,
                is_terminal=False,
            )

        normalized = np.asarray(
            action["action"],
            dtype=np.float32,
        )

        if normalized.shape != (6,):
            raise ValueError(
                f"Expected action shape (6,), got {normalized.shape}"
            )

        normalized = np.clip(normalized, -1.0, 1.0)

        # Use small delta-joint commands instead of jumping directly across
        # the full actuator range.
        target = np.asarray(
            self._env.data.ctrl,
            dtype=np.float32,
        ).copy()

        target[:5] += normalized[:5] * self._joint_delta

        low = self._env.action_space.low
        high = self._env.action_space.high
        target[:5] = np.clip(target[:5], low[:5], high[:5])

        # Last action component controls the gripper:
        # negative = close, positive = open, near zero = keep current command.
        if normalized[5] < -self._gripper_threshold:
            target[5] = low[5]
        elif normalized[5] > self._gripper_threshold:
            target[5] = high[5]

        raw, reward, terminated, truncated, info = self._env.step(
            target.astype(np.float32)
        )

        reward = self._shape_reward(
            base_reward=float(reward),
            info=info,
            terminated=bool(terminated),
        )

        self._done = bool(terminated or truncated)
        self._last_info = info

        return self._convert(
            raw=raw,
            info=info,
            reward=reward,
            is_first=False,
            is_last=self._done,
            is_terminal=bool(terminated),
        )


    def _shape_reward(
        self,
        base_reward: float,
        info: dict[str, Any],
        terminated: bool,
    ) -> np.float32:
        current_distance = float(
            info["object_to_grasp_distance"]
        )
        current_object_z = float(info["object_z"])
        grasped = bool(
            getattr(self._env, "_grasped", False)
        )

        if self._previous_distance is None:
            reach_progress = 0.0
        else:
            reach_progress = float(
                np.clip(
                    self._previous_distance
                    - current_distance,
                    -0.02,
                    0.02,
                )
            )

        if self._previous_object_z is None or not grasped:
            lift_progress = 0.0
        else:
            lift_progress = float(
                np.clip(
                    current_object_z
                    - self._previous_object_z,
                    -0.02,
                    0.02,
                )
            )

        newly_grasped = (
            grasped and not self._previous_grasped
        )
        dropped = (
            self._previous_grasped and not grasped
        )
        success = bool(info.get("success", False))
        failure = bool(terminated and not success)

        # Base environment gives +1 on success. Add another +9
        # so that total success reward becomes approximately +10.
        shaped_reward = (
            base_reward
            + 2.0 * reach_progress
            + 20.0 * lift_progress
            + 1.0 * float(newly_grasped)
            + 9.0 * float(success)
            - 1.0 * float(dropped)
            - 2.0 * float(failure)
        )

        info["reward_reach"] = 2.0 * reach_progress
        info["reward_lift"] = 20.0 * lift_progress
        info["reward_grasp"] = float(newly_grasped)
        info["reward_success"] = 10.0 * float(success)
        info["reward_drop"] = -1.0 * float(dropped)
        info["reward_failure"] = -2.0 * float(failure)
        info["reward_total"] = shaped_reward

        self._previous_distance = current_distance
        self._previous_object_z = current_object_z
        self._previous_grasped = grasped

        return np.float32(shaped_reward)

    def _convert(
        self,
        raw: dict[str, np.ndarray],
        info: dict[str, Any],
        reward: float,
        is_first: bool,
        is_last: bool,
        is_terminal: bool,
    ) -> dict[str, Any]:
        robot_state = np.asarray(
            raw["observation.state"],
            dtype=np.float32,
        )

        if self._mode == "state":
            vector = np.concatenate(
                [
                    robot_state,
                    np.asarray(
                        info["object_position"],
                        dtype=np.float32,
                    ),
                    np.asarray(
                        info["grasp_position"],
                        dtype=np.float32,
                    ),
                    np.asarray(
                        [info["object_to_grasp_distance"]],
                        dtype=np.float32,
                    ),
                ]
            )
        else:
            vector = robot_state

        observation: dict[str, Any] = {
            "vector": vector.astype(np.float32),
            "reward": np.float32(reward),
            "is_first": bool(is_first),
            "is_last": bool(is_last),
            "is_terminal": bool(is_terminal),
        }

        if self._mode == "vision":
            observation["image"] = np.asarray(
                raw["observation.images.front"],
                dtype=np.uint8,
            )

        return observation

    def render(self) -> np.ndarray:
        return self._env.render()

    def close(self) -> None:
        self._env.close()
