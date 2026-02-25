from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List, Any, Optional

import pybullet as p


@dataclass
class SpecimenMeta:
    creature: str
    time_step: float
    frames: int


class TrajectoryRecorder:
    def __init__(self, body_id: int, creature_name: str, time_step: float):
        self.body_id = int(body_id)
        self.creature_name = str(creature_name)
        self.time_step = float(time_step)
        self.frames: List[Dict[str, Any]] = []

    def capture(self):
        frame: Dict[str, Any] = {}

        base_pos, base_orn = p.getBasePositionAndOrientation(self.body_id)
        frame["base"] = {"pos": [float(x) for x in base_pos], "orn": [float(x) for x in base_orn]}

        n = p.getNumJoints(self.body_id)
        links = []
        for j in range(n):
            st = p.getLinkState(self.body_id, j, computeForwardKinematics=True)
            # st[4]=worldLinkFramePosition, st[5]=worldLinkFrameOrientation
            pos = st[4]
            orn = st[5]
            links.append({"index": int(j), "pos": [float(x) for x in pos], "orn": [float(x) for x in orn]})
        frame["links"] = links

        self.frames.append(frame)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "meta": {
                "creature": self.creature_name,
                "time_step": self.time_step,
                "frames": len(self.frames),
            },
            "trajectory": self.frames,
        }

    def save_json(self, path: str):
        data = self.to_dict()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)