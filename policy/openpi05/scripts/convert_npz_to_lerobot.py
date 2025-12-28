"""
Custom script to convert a local dataset of npz + json episodes into LeRobot format.
"""

import json
import shutil
from pathlib import Path

import numpy as np
import tyro
from tqdm import tqdm

from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

# --- 1. CONFIGURE YOUR DATASET ---
REPO_NAME = "windyhu/ur5e_slab_pick_up_task"   # 最终会保存到 $HF_LEROBOT_HOME/windyhu/panda_slab_pick_up_task
ROBOT_TYPE = "UR5e"
FPS = 30

ZERO_IMG = np.zeros((256, 256, 3), dtype=np.uint8)

def main(data_dir: str, *, push_to_hub: bool = False):
    """
    Args:
        data_dir: the root that contains "episodes/" (each episode has steps.npz + meta.json)
        push_to_hub: if True, push to HF hub
    """
    # --- Clean up any existing dataset at output path ---
    output_path = HF_LEROBOT_HOME / REPO_NAME      # << 关键：清理真正的输出目录
    if output_path.exists():
        print(f"Removing existing dataset at: {output_path}")
        shutil.rmtree(output_path)

    # --- Create LeRobot dataset schema ---
    print("Creating LeRobot dataset structure...")
    dataset = LeRobotDataset.create(
        repo_id=REPO_NAME,
        robot_type=ROBOT_TYPE,
        fps=FPS,
        features={
            "image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image": {               
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (8,),
                "names": ["eef_x","eef_y","eef_z","eef_rx","eef_ry","eef_rz","gripper_state","gripper_state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["d_x","d_y","d_z","d_roll","d_pitch","d_yaw","gripper"],
            },
        },
        image_writer_threads=16,
        image_writer_processes=4,
    )

    # --- 2. Convert episodes ---
    episodes_dir = Path(data_dir) / "episodes"
    episode_paths = sorted(list(episodes_dir.glob("*/")))
    assert len(episode_paths) > 0, f"No episode directories found in {episodes_dir}"

    print(f"Found {len(episode_paths)} episodes. Starting conversion...")
    for ep_path in tqdm(episode_paths, desc="Converting episodes"):
        with open(ep_path / "meta.json", "r") as f:
            meta = json.load(f)
        steps = np.load(ep_path / "steps.npz")

        language_instruction = meta.get("language_instruction", "")
        T = int(meta["episode_len"])

        has_wrist = "observation.wrist_image" in steps

        img = steps["observation.image"]
        assert img.ndim == 4 and img.shape[1:3] == (256, 256), f"expect [T,256,256,3], got {img.shape}"
        st = steps["observation.state"]
        act = steps["action"]
        assert st.shape[0] == T and st.shape[1] == 8, f"state shape must be [T,8], got {st.shape}"
        assert act.shape[0] == T and act.shape[1] == 7, f"actions shape must be [T,7], got {act.shape}"

        for t in range(T):
            frame_data = {
                "image": img[t],
                "wrist_image": steps["observation.wrist_image"][t] if has_wrist else ZERO_IMG,
                "state": st[t],
                "actions": act[t],
                "task": language_instruction,
            }
            dataset.add_frame(frame_data)

        dataset.save_episode()

    print(f"Conversion complete! Wrote to {output_path}")

    # --- 3. (Optional) push to hub ---
    if push_to_hub:
        print(f"Pushing dataset to Hugging Face Hub: {REPO_NAME}")
        dataset.push_to_hub(
            tags=["custom", ROBOT_TYPE, "teleop"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )
        print("Push to Hub complete!")


if __name__ == "__main__":
    tyro.cli(main)
