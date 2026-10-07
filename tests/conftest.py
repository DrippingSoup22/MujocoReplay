"""Shared test helpers: a small model, and recordings of it built in memory."""

import mujoco
import numpy as np
import pytest

from mujoco_replay.recording import Recording

# A floor, two lights, a static post, a door on a hinge, and a robot: a free
# root body with a painted shape and a hidden one, and an arm on a hinge. The
# arm's hinge is unnamed and the spot light aims at the robot, two cases the
# composition must handle.
SMALL_MODEL = """
<mujoco model="small">
  <asset>
    <material name="paint" rgba="0.2 0.4 0.9 1"/>
  </asset>
  <worldbody>
    <light name="sun" pos="0 0 4" dir="0 0 -1"/>
    <light name="spot" pos="2 -2 3" mode="targetbody" target="robot"/>
    <geom name="floor" type="plane" size="3 3 0.1" rgba="0.9 0.9 0.9 1"/>
    <body name="post" pos="1.5 0 0.3">
      <geom name="post" type="cylinder" size="0.05 0.3" rgba="0.5 0.3 0.1 1"/>
    </body>
    <body name="door" pos="-1.5 0 0.5">
      <joint name="door_hinge" type="hinge" axis="0 0 1"/>
      <geom name="door" type="box" size="0.3 0.02 0.5" rgba="0.6 0.6 0.2 1"/>
    </body>
    <body name="robot" pos="0 0 0.5">
      <freejoint/>
      <geom name="torso" type="box" size="0.2 0.1 0.05" material="paint"/>
      <geom name="bumper" type="sphere" size="0.12" rgba="0 0 0 0"/>
      <body name="arm" pos="0.2 0 0">
        <joint type="hinge" axis="0 1 0"/>
        <geom name="forearm" type="capsule" fromto="0 0 0 0.3 0 0" size="0.03"
              rgba="0.9 0.3 0.1 1"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="door_motor" joint="door_hinge"/>
  </actuator>
</mujoco>
"""


@pytest.fixture
def small_model() -> str:
    return SMALL_MODEL


@pytest.fixture
def make_recording():
    """A function that builds a recording with random, valid poses."""

    def make(
        frames: int = 2,
        worlds: int = 3,
        model_xml: str = SMALL_MODEL,
        seed: int = 0,
        **fields,
    ) -> Recording:
        model = mujoco.MjModel.from_xml_string(model_xml)
        rng = np.random.default_rng(seed)
        qpos = model.qpos0 + rng.normal(0.0, 0.3, (frames, worlds, model.nq))
        for row in qpos.reshape(-1, model.nq):
            mujoco.mj_normalizeQuat(model, row)
        return Recording(model_xml, 0.02, qpos, **fields)

    return make
