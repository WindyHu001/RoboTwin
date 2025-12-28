#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
pi0 + LIBERO evaluation (official suites or custom BDDL/init pairs)

Usage (official suite):
  python scripts/eval_libero_pi0_custom.py --task_suite_name libero_spatial --num_trials_per_task 10

Usage (custom tasks):
  python scripts/eval_libero_pi0_custom.py --task_suite_name custom \
      --custom_tasks '[{"bddl":"/ABS/PATH/task1.bddl","init":"/ABS/PATH/task1.pruned_init","desc":"pick up ..."}]' \
      --num_trials_per_task 5

Model server:
  --host 0.0.0.0 --port 8000

Notes:
- Actions are 7D: [dx,dy,dz,rx,ry,rz,gripper], with a dummy "do-nothing" [-1] gripper at start.
- Images are rotated 180° (flip H+V) and resized+letterboxed to `resize_size` using openpi_client.image_tools.
"""

import collections
import dataclasses
import json
import logging
import math
import pathlib
from typing import List, Dict, Any
import os,sys

import imageio
import numpy as np
import tqdm
import tyro

from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from datetime import datetime
from scipy.spatial.transform import Rotation as R

# pi0 client utils
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
custom_code_dir = "/home/crosslab/vla/LIBERO/libero/libero_custom_spatial_finetune/code"
print(custom_code_dir)
if custom_code_dir not in sys.path:
    sys.path.insert(0, custom_code_dir)
import objects
import scene


# ------------------------------ defaults / constants ------------------------------
LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]   # 7D: keep gripper closed by default
LIBERO_ENV_RESOLUTION = 256                # rendering resolution used for training data


# ------------------------------ CLI args ------------------------------
@dataclasses.dataclass
class Args:
    # Model server (pi0) -----------------------------------------------------------
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224              # pi0 input resolution (post letterbox)
    replan_steps: int = 5               # predict chunks; execute first K each loop

    # Which tasks to run -----------------------------------------------------------
    task_suite_name: str = "custom"   # libero_spatial | libero_object | libero_goal | libero_10 | libero_90 | custom
    num_trials_per_task: int = 50             # rollouts per task

    # Custom tasks (used when task_suite_name == "custom")
    # JSON string: list of {"bddl": "...", "init": "...", "desc": "..."}
    custom_tasks: str = json.dumps([{
        "bddl": "/home/crosslab/vla/LIBERO/libero/libero_custom_spatial_finetune/code/custom_pddl/KITCHEN_DEMO_SCENE_libero_demo_behaviors.bddl",
        "init": "/home/crosslab/vla/LIBERO/libero/libero_custom_spatial_finetune/code/custom_pddl/KITCHEN_DEMO_SCENE_libero_demo_behaviors.pruned_init",
        "desc": "Pick up the blue glass slab by staying vertical and place it on the gray stand."
        # "desc":"First, pick up the glass slab placed on the ground, and then move it to the rack on the left."
    }])

    # Sim / control ---------------------------------------------------------------
    num_steps_wait: int = 10            # let objects settle before acting
    max_steps_override: int = -1        # if >0, override auto max-steps per suite
    camera_resolution: int = LIBERO_ENV_RESOLUTION
    seed: int = 7

    # Logging / outputs -----------------------------------------------------------
    video_out_path: str = "data/libero/videos"
    log_level: str = "INFO"


# ------------------------------ helpers ------------------------------
# def _quat2axisangle(quat: np.ndarray) -> np.ndarray:
#     """robosuite-style quat (x,y,z,w) to axis-angle (3,)"""
#     q = quat.copy()
#     q[3] = np.clip(q[3], -1.0, 1.0)
#     den = np.sqrt(max(1e-12, 1.0 - q[3] * q[3]))
#     if math.isclose(den, 0.0):
#         return np.zeros(3, dtype=np.float32)
#     return (q[:3] * 2.0 * math.acos(q[3]) / den).astype(np.float32)
def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den

def _make_env_from_suite_task(task, resolution: int, seed: int):
    """Create LIBERO env for an official benchmark task."""
    task_bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(
        bddl_file_name=str(task_bddl),
        camera_heights=resolution,
        camera_widths=resolution,
    )
    env.seed(seed)
    return env, task.language


def _make_env_from_custom(bddl_path: str, resolution: int, seed: int, **kwargs):
    """Create LIBERO env for a custom BDDL file."""
    env = OffScreenRenderEnv(
        bddl_file_name=bddl_path,
        camera_heights=resolution,
        camera_widths=resolution,
        robots=["Panda"],
        **kwargs,  # e.g., robots=["Panda"], camera_names=...
    )
    env.seed(seed)
    return env


def _preprocess_obs(obs: Dict[str, Any], resize_size: int):
    """Prepare pi0 inputs + replay-frame (uint8)."""
    # Rotate 180° to match training preprocessing
    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wrist = np.ascontiguousarray(obs.get("robot0_eye_in_hand_image", img)[::-1, ::-1])

    img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, resize_size, resize_size))
    wrist = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist, resize_size, resize_size))

    # State = [eef_pos(3), eef_axisangle(3), gripper_qpos(1 or 2)]
    state = np.concatenate(
        (
            # obs["robot0_eef_pos"].astype(np.float32),
            _quat2axisangle(obs["robot0_eef_quat"].astype(np.float32)),
            np.atleast_1d(obs["robot0_gripper_qpos"]).astype(np.float32),
        )
    ).astype(np.float32)
    return img, wrist, state


def _preprocess_sim(sim, robot, obs: Dict[str, Any], resize_size: int):
    """Prepare pi0 inputs + replay-frame (uint8), state aligned to _get_eef_state8.
       State = [eef_pos(3), eef_rotvec(3), 0, 0]  (读取自 sim，与 _get_eef_state8 一致)
    """
    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wrist = np.ascontiguousarray(obs.get("robot0_eye_in_hand_image", img)[::-1, ::-1])

    img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, resize_size, resize_size))
    wrist = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist, resize_size, resize_size))

    pos = sim.data.site_xpos[robot.eef_site_id].astype(np.float32).copy()
    R_now = sim.data.site_xmat[robot.eef_site_id].reshape(3, 3)
    rotvec = R.from_matrix(R_now).as_rotvec().astype(np.float32)

    grip_dummy = np.array([0.0, 0.0], dtype=np.float32)

    state = np.concatenate([pos, rotvec, grip_dummy]).astype(np.float32)
    return img, wrist, state

def _infer_chunk(client, img, wrist, state, prompt: str, replan_steps: int) -> List[List[float]]:
    """Query pi0 server; return a list of actions (T x 7)."""
    element = {
        "observation/image": img,
        "observation/wrist_image": wrist,
        "observation/state": state,
        "prompt": prompt,
    }
    # import pdb;pdb.set_trace()

    out = client.infer(element)
    actions = out["actions"]  # expect iterable of 7D actions
    if len(actions) < replan_steps:
        raise RuntimeError(
            f"Policy returned {len(actions)} steps but replan_steps={replan_steps}."
        )
    return [list(map(float, a)) for a in actions[:replan_steps]]


def _suite_max_steps(name: str) -> int:
    """Longest training demo per suite (with a small margin)."""
    if name == "libero_spatial":
        return 220
    if name == "libero_object":
        return 280
    if name == "libero_goal":
        return 300
    if name == "libero_10":
        return 520
    if name == "libero_90":
        return 400
    return 1000  # fallback


# ------------------------------ main eval ------------------------------
def eval_libero_pi0(args: Args) -> None:
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    np.random.seed(args.seed)
    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    # --- Task source: official suite vs custom ---
    if args.task_suite_name != "custom":
        bench = benchmark.get_benchmark_dict()[args.task_suite_name]()
        tasks = [{"_suite_task": bench.get_task(i)} for i in range(bench.n_tasks)]
        task_iter = tqdm.tqdm(range(len(tasks)))
        max_steps = args.max_steps_override if args.max_steps_override > 0 else _suite_max_steps(args.task_suite_name)
    else:
        try:
            tasks = json.loads(args.custom_tasks)
            assert isinstance(tasks, list) and all(
                all(k in t for k in ("bddl", "init", "desc")) for t in tasks
            ), "custom_tasks must be a JSON list of {bddl, init, desc}"
        except Exception as e:
            raise ValueError(f"--custom_tasks parse error: {e}")
        task_iter = tqdm.tqdm(range(len(tasks)))
        max_steps = args.max_steps_override if args.max_steps_override > 0 else 1000  # reasonable default

    total_ep = total_success = 0

    for ti in task_iter:
        if args.task_suite_name != "custom":
            suite_task = tasks[ti]["_suite_task"]
            task_desc = suite_task.language
            init_states = bench.get_task_init_states(ti)
            env, _ = _make_env_from_suite_task(suite_task, args.camera_resolution, args.seed)
            n_eps = min(args.num_trials_per_task, len(init_states))
        else:
            ct = tasks[ti]
            task_desc = str(ct["desc"])
            # load pickled init states
            with open(ct["init"], "rb") as f:
                init_states = json.load(f) if ct["init"].endswith(".json") else __import__("pickle").load(f)
            env = _make_env_from_custom(ct["bddl"], args.camera_resolution, args.seed)
            n_eps = min(args.num_trials_per_task, len(init_states))

        task_success = task_episodes = 0

        for ep_idx in range(n_eps):
            # Reset + set deterministic initial state
            obs = env.reset()
            obs = env.set_init_state(init_states[ep_idx])
            # try:
            #     cam_id = env.sim.model.camera_name2id("agentview")
            #     env.sim.model.cam_pos[cam_id] = [1.0, 0.25, 0.6]
            #     env.sim.model.cam_quat[cam_id] = [0.5938, 0.3839, 0.3839, 0.5938]
            #     obs, _, _, _ = env.step(np.zeros(7))
            #     print("   Camera pose set and view refreshed.")
            # except Exception as e:
            #     print(f"[WARNING] Failed to modify camera pose: {e}")
            # Warmup: let objects settle
            t = 0
            replay = []
            action_plan = collections.deque()

            while t < (max_steps + args.num_steps_wait):
                try:
                    if t < args.num_steps_wait:
                        obs, _, done, _ = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        if done:
                            break
                        continue

                    # Preprocess obs
                    # img, wrist, state = _preprocess_obs(obs, args.resize_size)
                    img, wrist, state = _preprocess_sim(env.sim, env.robots[0], obs, args.resize_size)

                    replay.append(img)  # store processed (letterboxed, uint8) frame

                    # Replan if needed
                    if not action_plan:
                        chunk = _infer_chunk(client, img, wrist, state, task_desc, args.replan_steps)
                        action_plan.extend(chunk)

                    action = action_plan.popleft()
                    # action[3:6] = [0.0, 0.0, 0.0]
                    obs, reward, done, info = env.step(action)

                    if done:
                        task_success += 1
                        total_success += 1
                        break

                    t += 1

                except Exception as e:
                    logging.error(f"[Task {ti} ep {ep_idx}] exception: {e}")
                    break

            task_episodes += 1
            total_ep += 1

            # Save replay (10 fps)
            try:
                suffix = "success" if done else "failure"
                task_tag = task_desc.replace(" ", "_")[:128]

                now = datetime.now()
                date_dir = now.strftime("%m-%d")       
                time_prefix = now.strftime("%H-%M")    

                out_dir = pathlib.Path(args.video_out_path) / date_dir
                out_dir.mkdir(parents=True, exist_ok=True)

                filename = f"{time_prefix}_rollout_task{ti}_ep{ep_idx+1}_{task_tag}_{suffix}.mp4"
                out_file = out_dir / filename

                if replay:
                    imageio.mimwrite(out_file, [np.asarray(f) for f in replay], fps=10)
            except Exception as e:
                logging.warning(f"Failed to write video: {e}")


            logging.info(f"[Task {ti}] ep={task_episodes} success={bool(done)} "
                         f"total={total_success}/{total_ep} ({100.0*total_success/max(total_ep,1):.1f}%)")

        logging.info(f"[Task {ti}] task_success_rate={task_success/max(task_episodes,1):.3f}")

    logging.info(f"Final: success={total_success}/{total_ep} ({100.0*total_success/max(total_ep,1):.1f}%), episodes={total_ep}")


# ------------------------------ entry ------------------------------
if __name__ == "__main__":
    tyro.cli(eval_libero_pi0)
