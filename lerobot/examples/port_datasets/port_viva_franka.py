#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Convert lossless Viva-La-Franka SmolVLA episodes to LeRobotDataset v3.

Run this offline on the training workstation, not in the robot recording
process. The source action is an 8-D contract: seven normalized realized joint
deltas followed by an absolute binary gripper target. It is deliberately not
renamed or reinterpreted as joint velocity.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from PIL import Image

FPS = 15
CAMERA_SOURCES = {
    "wrist": "hand_camera",
    "top": "varied_camera_1",
    "right": "varied_camera_2",
}
STATE_NAMES = ["q0", "q1", "q2", "q3", "q4", "q5", "q6", "gripper"]
ACTION_NAMES = [
    "joint_delta_0",
    "joint_delta_1",
    "joint_delta_2",
    "joint_delta_3",
    "joint_delta_4",
    "joint_delta_5",
    "joint_delta_6",
    "gripper_target",
]


def dataset_features(width: int, height: int) -> dict[str, dict[str, Any]]:
    features: dict[str, dict[str, Any]] = {
        "observation.state": {
            "dtype": "float32",
            "shape": (8,),
            "names": {"axes": STATE_NAMES},
        },
        "action": {
            "dtype": "float32",
            "shape": (8,),
            "names": {"axes": ACTION_NAMES},
        },
        "layout_id": {"dtype": "string", "shape": (1,), "names": None},
        "source_episode": {"dtype": "string", "shape": (1,), "names": None},
    }
    for role in CAMERA_SOURCES:
        features[f"observation.images.{role}"] = {
            "dtype": "video",
            "shape": (height, width, 3),
            "names": ["height", "width", "channels"],
        }
    return features


def discover_episodes(project_root: Path) -> list[Path]:
    episodes = []
    for episode in sorted(project_root.glob("task_*/demo_*")):
        if not episode.is_dir() or episode.name.endswith(".inprogress"):
            continue
        metadata_path = episode / "metadata.json"
        try:
            metadata = json.loads(metadata_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if metadata.get("status") == "complete" and metadata.get("profile") == "smolvla":
            episodes.append(episode.resolve())
    return episodes


def load_selection_manifest(path: Path, project_root: Path) -> list[Path]:
    episodes = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if row.get("episode_relpath"):
                    episode = project_root / row["episode_relpath"]
                else:
                    episode = Path(row["episode"]).expanduser()
                episodes.append(episode.resolve())
            except (json.JSONDecodeError, KeyError) as exc:
                raise ValueError(f"invalid selection manifest line {line_number}: {exc}") from exc
    return episodes


def load_episode_contract(episode: Path) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    metadata = json.loads((episode / "metadata.json").read_text())
    source = json.loads((episode / "smolvla_episode.json").read_text())
    if metadata.get("status") != "complete" or metadata.get("profile") != "smolvla":
        raise ValueError(f"episode is not an accepted SmolVLA source: {episode}")
    if source.get("format") != "viva_franka_smolvla_source" or int(source.get("fps", -1)) != FPS:
        raise ValueError(f"unexpected source contract: {episode}")
    if not str(metadata.get("layout_id", "")).strip():
        raise ValueError(f"episode has no layout id: {episode}")
    with h5py.File(episode / "trajectory.h5", "r") as h5:
        state = np.asarray(h5["lerobot/observation_state"][:], dtype=np.float32)
        action = np.asarray(h5["lerobot/action"][:], dtype=np.float32)
        if h5.attrs.get("export_profile", "") != "smolvla":
            raise ValueError(f"trajectory export profile is not smolvla: {episode}")
    if state.ndim != 2 or state.shape[1] != 8 or action.shape != state.shape:
        raise ValueError(f"invalid state/action shapes {state.shape}/{action.shape}: {episode}")
    if len(state) != int(source.get("num_frames", -1)):
        raise ValueError(f"frame count differs from source manifest: {episode}")
    if not np.all(np.isfinite(state)) or not np.all(np.isfinite(action)):
        raise ValueError(f"state/action contains non-finite values: {episode}")
    for role, folder in CAMERA_SOURCES.items():
        camera_dir = episode / "recordings" / "frames" / folder
        if len(list(camera_dir.glob("*.jpg"))) != len(state):
            raise ValueError(f"{role} image count differs from trajectory: {episode}")
    return metadata, state, action


def read_rgb(path: Path, width: int, height: int) -> np.ndarray:
    with Image.open(path) as image:
        image = image.convert("RGB")
        if image.size != (width, height):
            image = image.resize((width, height), Image.Resampling.LANCZOS)
        return np.asarray(image, dtype=np.uint8)


def episode_frames(episode: Path, width: int, height: int) -> Iterator[dict[str, Any]]:
    metadata, state, action = load_episode_contract(episode)
    task = str(metadata["prompt"]).strip()
    layout_id = str(metadata["layout_id"]).strip()
    for index in range(len(state)):
        frame: dict[str, Any] = {
            "task": task,
            "observation.state": state[index],
            "action": action[index],
            "layout_id": layout_id,
            "source_episode": str(episode),
        }
        for role, folder in CAMERA_SOURCES.items():
            path = episode / "recordings" / "frames" / folder / f"{index:09d}.jpg"
            frame[f"observation.images.{role}"] = read_rgb(path, width, height)
        yield frame


def validate_plan(episodes: list[Path]) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    total_frames = 0
    tasks: set[str] = set()
    for episode in episodes:
        metadata, state, _ = load_episode_contract(episode)
        counts[str(metadata["layout_id"])] += 1
        total_frames += len(state)
        tasks.add(str(metadata["prompt"]).strip())
    return {
        "episodes": len(episodes),
        "frames": total_frames,
        "hours": total_frames / FPS / 3600,
        "layouts": dict(sorted(counts.items())),
        "tasks": sorted(tasks),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--image-width", type=int, default=640)
    parser.add_argument("--image-height", type=int, default=360)
    parser.add_argument(
        "--streaming-encoding",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="encode MP4 while frames are read; avoids a large temporary PNG tree",
    )
    parser.add_argument("--encoder-threads", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--private", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    if args.image_width <= 0 or args.image_height <= 0:
        raise SystemExit("image dimensions must be positive")
    episodes = (
        load_selection_manifest(args.selection_manifest, args.project_root)
        if args.selection_manifest is not None
        else discover_episodes(args.project_root)
    )
    if not episodes:
        raise SystemExit("no accepted SmolVLA episodes found")
    summary = validate_plan(episodes)
    print(json.dumps(summary, indent=2))
    if args.dry_run:
        return 0
    if args.output_root.exists():
        raise SystemExit(f"refusing to overwrite existing output root: {args.output_root}")

    from lerobot.datasets import LeRobotDataset

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        root=args.output_root,
        robot_type="franka_fr3",
        fps=FPS,
        features=dataset_features(args.image_width, args.image_height),
        use_videos=True,
        streaming_encoding=args.streaming_encoding,
        encoder_threads=args.encoder_threads,
        image_writer_threads=8 if not args.streaming_encoding else 0,
    )
    for index, episode in enumerate(episodes, 1):
        logging.info("Converting %d/%d: %s", index, len(episodes), episode)
        for frame in episode_frames(episode, args.image_width, args.image_height):
            dataset.add_frame(frame)
        dataset.save_episode()
    dataset.finalize()
    if args.push_to_hub:
        dataset.push_to_hub(tags=["franka", "fr3", "smolvla"], private=args.private)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
