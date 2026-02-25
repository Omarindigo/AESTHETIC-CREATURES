from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import gymnasium as gym
import pybullet as p
import pybullet_data

from .utils import SimConfig
from .creatures import get_creature_spec, build_creature
from .controllers import OscillatorController, OscillatorParams


@dataclass
class EnvConfig:
    creature: str = "starfish"
    gui: bool = False
    episode_seconds: float = 10.0
    sim: SimConfig = SimConfig()
    ground: bool = True


class AestheticCreatureEnv(gym.Env):
    metadata = {"render_modes": ["human", "none"]}

    def __init__(self, cfg: EnvConfig):
        super().__init__()
        self.cfg = cfg

        # Action controls oscillator params (amp, freq, phase_stride, bias, mode blend).
        # Keeping it continuous makes it RL-ready later.
        self.action_space = gym.spaces.Box(
            low=np.array([0.0, 0.1, 0.0, -2.0, 0.0], dtype=np.float32),
            high=np.array([2.0, 4.0, 2.5, 2.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        # Observation: base (pos, orn, lin vel, ang vel) + joint (pos, vel) for each joint.
        # We don’t encode geometry here; geometry is fixed by creature spec, which matches your “shape determines motion” idea.
        self._max_joints = 16
        obs_dim = 3 + 4 + 3 + 3 + (self._max_joints * 2)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        self.client_id = None
        self.body_id = None
        self.joint_count = 0
        self.controller: Optional[OscillatorController] = None
        self.t = 0.0
        self.max_steps = int(self.cfg.episode_seconds / self.cfg.sim.time_step)

    def _connect(self):
        if self.client_id is not None:
            return
        mode = p.GUI if self.cfg.gui else p.DIRECT
        self.client_id = p.connect(mode)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        p.setGravity(*self.cfg.sim.gravity)
        p.setTimeStep(self.cfg.sim.time_step)

        if self.cfg.gui:
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)

    def _disconnect(self):
        if self.client_id is not None:
            p.disconnect(self.client_id)
            self.client_id = None

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self._connect()
        p.resetSimulation()

        p.setGravity(*self.cfg.sim.gravity)
        p.setTimeStep(self.cfg.sim.time_step)

        if self.cfg.ground:
            p.loadURDF("plane.urdf")

        spec = get_creature_spec(self.cfg.creature)
        self.body_id, _ = build_creature(spec, base_pos=(0.0, 0.0, 0.65))

        self.joint_count = p.getNumJoints(self.body_id)
        joint_indices = list(range(self.joint_count))

        params = OscillatorController.from_defaults(spec.controller_defaults)
        self.controller = OscillatorController(self.body_id, joint_indices, params)
        self.controller.reset()

        # Small random initial perturbation so runs aren’t identical.
        for j in joint_indices:
            p.resetJointState(self.body_id, j, targetValue=float(self.np_random.uniform(-0.1, 0.1)))

        self.t = 0.0
        obs = self._get_obs()
        info = {}
        return obs, info

    def _get_obs(self) -> np.ndarray:
        base_pos, base_orn = p.getBasePositionAndOrientation(self.body_id)
        base_lin, base_ang = p.getBaseVelocity(self.body_id)

        joint_pos = np.zeros((self._max_joints,), dtype=np.float32)
        joint_vel = np.zeros((self._max_joints,), dtype=np.float32)

        for j in range(min(self.joint_count, self._max_joints)):
            js = p.getJointState(self.body_id, j)
            joint_pos[j] = float(js[0])
            joint_vel[j] = float(js[1])

        obs = np.concatenate(
            [
                np.array(base_pos, dtype=np.float32),
                np.array(base_orn, dtype=np.float32),
                np.array(base_lin, dtype=np.float32),
                np.array(base_ang, dtype=np.float32),
                joint_pos,
                joint_vel,
            ],
            axis=0,
        )
        return obs

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        amp, freq, phase_stride, bias, mode_mix = action.tolist()

        # Update controller params each step.
        if self.controller is not None:
            self.controller.params.amp = float(amp)
            self.controller.params.freq = float(freq)
            self.controller.params.phase_stride = float(phase_stride)
            self.controller.params.bias = float(bias)
            self.controller.params.mode = 0 if float(mode_mix) < 0.5 else 1

            self.controller.step(self.cfg.sim.time_step)

        for _ in range(int(self.cfg.sim.substeps)):
            p.stepSimulation()

        self.t += float(self.cfg.sim.time_step)
        obs = self._get_obs()

        # “Reward” is deliberately aesthetic-neutral; we keep env usable without optimizing yet.
        # For now: small alive reward so RL can run; later you can replace with novelty/energy/curation metrics.
        reward = 0.1

        terminated = False
        truncated = (int(self.t / self.cfg.sim.time_step) >= self.max_steps)

        info: Dict[str, Any] = {"t": self.t}
        return obs, reward, terminated, truncated, info

    def render(self):
        # GUI handled by pybullet; nothing needed here.
        return None

    def close(self):
        self._disconnect()