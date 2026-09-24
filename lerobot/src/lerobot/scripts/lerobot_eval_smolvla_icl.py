#!/usr/bin/env python
"""为 SmolVLA-ICL 显式注册 test Demo，并运行在线 LIBERO rollout。"""

import json
import logging
import math
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from pprint import pformat
from typing import Any

import torch

from lerobot.configs import parser
from lerobot.configs.eval import EvalPipelineConfig
from lerobot.envs import close_envs, make_env, make_env_pre_post_processors
from lerobot.policies import make_policy, make_pre_post_processors
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla_icl.components.demo_alignment import load_smolvla_siglip
from lerobot.policies.smolvla_icl.data.libero_manifest import (
    LiberoDataManifest,
    libero_task_index,
)
from lerobot.policies.smolvla_icl.data.state import DemoStateNormalizer
from lerobot.policies.smolvla_icl.data.train_stats import TrainStatsArtifact
from lerobot.policies.smolvla_icl.evaluation import (
    AlignmentTraceCollector,
    TestDemoProvider,
    select_test_demos,
    summarize_demo_swap,
)
from lerobot.scripts.lerobot_eval import rollout
from lerobot.utils.constants import OBS_STATE
from lerobot.utils.device_utils import get_safe_torch_device
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import init_logging

logger = logging.getLogger(__name__)


@dataclass
class SmolVLAICLEvalPipelineConfig(EvalPipelineConfig):
    """在通用 LeRobot eval 配置上补充 ICL test 所需的数据身份。"""

    manifest_path: Path | None = None
    dataset_root: Path | None = None
    train_stats_path: Path | None = None
    matcher_model: str = "lerobot/smolvla_base"
    matcher_revision: str | None = None
    matcher_local_files_only: bool = False
    state_key: str = OBS_STATE
    video_backend: str | None = None
    tolerance_s: float = 1e-4
    query_window_replans: int = 4
    swap_demo_count: int = 1
    demo_seed: int = 42

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.policy is None or self.policy.type != "smolvla_icl":
            raise ValueError("专用 evaluator 只接受 SmolVLA-ICL checkpoint。")
        if self.env.type != "libero":
            raise ValueError("当前 SmolVLA-ICL 在线 test evaluator 只支持 LIBERO。")
        if self.manifest_path is None:
            configured = getattr(self.policy, "data_manifest_path", None)
            self.manifest_path = Path(configured) if configured else None
        if self.train_stats_path is None:
            configured = getattr(self.policy, "training_stats_path", None)
            self.train_stats_path = Path(configured) if configured else None
        if self.manifest_path is None or self.train_stats_path is None or self.dataset_root is None:
            raise ValueError("必须提供 manifest_path、train_stats_path 和 dataset_root。")
        if self.seed is None:
            raise ValueError("在线 test 必须提供 seed，才能对 primary/swap Demo 做成对比较。")
        if self.eval.n_episodes < 1:
            raise ValueError("eval.n_episodes 必须大于 0。")
        if self.query_window_replans < 2 or self.swap_demo_count < 0:
            raise ValueError("query_window_replans 至少为 2，swap_demo_count 不能为负数。")
        if self.eval.recording:
            raise NotImplementedError("专用 ICL evaluator 首版不写 rollout Dataset。")

        # Matcher 和动作队列都保存单条 episode 状态；共享一个 Policy 并行
        # 跑多个 env/task 会相互覆盖，因此专用 evaluator 固定串行运行。
        self.eval.batch_size = 1
        self.eval.use_async_envs = False
        self.env.max_parallel_tasks = 1


def _validate_timing(
    cfg: SmolVLAICLEvalPipelineConfig,
    manifest: LiberoDataManifest,
) -> None:
    if not math.isclose(float(cfg.env.fps), manifest.fps, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(
            f"LIBERO env fps={cfg.env.fps} 与 Manifest fps={manifest.fps} 不一致。"
        )
    n_action_steps = cfg.policy.n_action_steps
    expected_hz = manifest.fps / n_action_steps
    expected_window = cfg.query_window_replans / expected_hz
    alignment = cfg.policy.demo_alignment
    if not math.isclose(alignment.alignment_hz, expected_hz, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(
            "Policy demo_alignment.alignment_hz 未绑定当前 action chunking："
            f"配置为 {alignment.alignment_hz}，应为 {expected_hz}。"
        )
    if not math.isclose(alignment.window_duration_s, expected_window, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(
            "Policy demo_alignment.window_duration_s 未绑定当前 action chunking："
            f"配置为 {alignment.window_duration_s}，应为 {expected_window}。"
        )


def _make_matcher(cfg: SmolVLAICLEvalPipelineConfig, device: torch.device):
    matcher_path = Path(cfg.matcher_model).expanduser()
    if cfg.matcher_revision is None and not matcher_path.exists():
        raise ValueError("Hub Matcher 必须显式提供 matcher_revision。")
    matcher_config = SmolVLAConfig.from_pretrained(
        cfg.matcher_model,
        revision=cfg.matcher_revision,
        local_files_only=cfg.matcher_local_files_only,
    )
    matcher_config.device = str(device)
    matcher = load_smolvla_siglip(
        cfg.matcher_model,
        device=device,
        freeze=True,
        config=matcher_config,
        revision=cfg.matcher_revision,
        local_files_only=cfg.matcher_local_files_only,
    )
    snapshot = (
        f"{cfg.matcher_model}@{cfg.matcher_revision}"
        if cfg.matcher_revision is not None
        else str(matcher_path.resolve())
    )
    return matcher, snapshot


def _episode_metrics(rollout_data: dict[str, torch.Tensor], seed: int) -> dict[str, Any]:
    done_index = int(torch.argmax(rollout_data["done"][0].to(torch.int64)))
    steps = done_index + 1
    rewards = rollout_data["reward"][0, :steps]
    successes = rollout_data["success"][0, :steps]
    return {
        "seed": seed,
        "steps": steps,
        "sum_reward": float(rewards.sum()),
        "max_reward": float(rewards.max()),
        "success": bool(successes.any()),
    }


def _aggregate_demo_result(episodes: list[dict[str, Any]]) -> dict[str, float]:
    return {
        "success_rate": sum(bool(item["success"]) for item in episodes) / len(episodes),
        "avg_sum_reward": sum(float(item["sum_reward"]) for item in episodes) / len(episodes),
        "avg_max_reward": sum(float(item["max_reward"]) for item in episodes) / len(episodes),
        "avg_steps": sum(int(item["steps"]) for item in episodes) / len(episodes),
    }


def evaluate_smolvla_icl(
    cfg: SmolVLAICLEvalPipelineConfig,
    *,
    envs: dict,
    policy: Any,
    env_preprocessor: Any,
    env_postprocessor: Any,
    preprocessor: Any,
    postprocessor: Any,
    manifest: LiberoDataManifest,
    demo_provider: TestDemoProvider,
    state_normalizer: DemoStateNormalizer,
    matcher: Any,
    matcher_snapshot: str,
    train_stats_fingerprint: str,
) -> dict[str, Any]:
    """按 task 注册 test Demo，并用相同 seed 执行 primary/swap rollouts。"""
    task_results: list[dict[str, Any]] = []
    primary_successes: list[bool] = []
    seeds = [int(cfg.seed) + offset for offset in range(cfg.eval.n_episodes)]
    test_task_indices = set(manifest.task_indices(split="test"))

    for suite, suite_envs in envs.items():
        for suite_task_id, env in suite_envs.items():
            task_index = libero_task_index(suite, suite_task_id)
            # Env factory 可以一次创建整个 suite；这里只 rollout Manifest 明确
            # 分到 test 的 unseen tasks，train/val tasks 不参与最终测试。
            if task_index not in test_task_indices:
                continue
            demos = select_test_demos(
                manifest,
                task_index=task_index,
                seed=cfg.demo_seed,
                swap_demo_count=cfg.swap_demo_count,
            )
            demo_results: list[dict[str, Any]] = []
            for demo_episode in demos:
                # LIBERO reset 会顺序轮换 init state。每条 swap Demo 都从 0
                # 重新开始，才能让相同 rollout seed 对应相同初始状态序列。
                env.set_attr("init_state_id", 0)
                demo = demo_provider.load(demo_episode)
                policy.set_demo(
                    demo.video,
                    demo.states,
                    demo.timestamps,
                    state_normalizer=state_normalizer,
                    matcher_image_key=manifest.image_key,
                    matcher_siglip=matcher,
                )
                episodes: list[dict[str, Any]] = []
                for seed in seeds:
                    # 同时固定模型的 flow-matching noise 与环境随机性，避免把
                    # 随机采样差异误记为 Demo-swap sensitivity。
                    set_seed(seed)
                    trace = AlignmentTraceCollector()
                    rollout_data = rollout(
                        env=env,
                        policy=policy,
                        env_preprocessor=env_preprocessor,
                        env_postprocessor=env_postprocessor,
                        preprocessor=preprocessor,
                        postprocessor=postprocessor,
                        seeds=[seed],
                        policy_step_callback=trace,
                    )
                    episode_metrics = _episode_metrics(rollout_data, seed)
                    episode_metrics["alignment_trace"] = trace.trace
                    episodes.append(episode_metrics)
                demo_results.append(
                    {
                        "demo_id": manifest.demo_id(demo_episode.episode_index),
                        "demo_episode_index": demo_episode.episode_index,
                        "episodes": episodes,
                        "aggregated": _aggregate_demo_result(episodes),
                    }
                )

            primary_successes.extend(bool(item["success"]) for item in demo_results[0]["episodes"])
            task_results.append(
                {
                    "suite": suite,
                    "suite_task_id": suite_task_id,
                    "task_index": task_index,
                    "task": demos[0].task,
                    "reserved_test_query_episodes": [
                        episode.episode_index
                        for episode in manifest.episodes
                        if episode.task_index == task_index
                        and episode.split == "test"
                        and episode.role == "query"
                    ],
                    "demo_results": demo_results,
                    "demo_swap_sensitivity": summarize_demo_swap(demo_results),
                }
            )

    if not primary_successes:
        raise ValueError("当前 env.task/task_ids 没有覆盖 Manifest 中的任何 test task。")

    return {
        "protocol": {
            "manifest_fingerprint": manifest.fingerprint,
            "train_stats_fingerprint": train_stats_fingerprint,
            "matcher_snapshot": matcher_snapshot,
            "n_episodes_per_demo": cfg.eval.n_episodes,
            "swap_demo_count_requested": cfg.swap_demo_count,
            "manifest_test_task_indices": sorted(test_task_indices),
            "evaluated_test_task_indices": [item["task_index"] for item in task_results],
            "query_source": "LIBERO simulator rollouts; Manifest test/query episodes remain reserved",
        },
        "overall": {
            "primary_demo_success_rate": sum(primary_successes) / len(primary_successes),
            "primary_demo_pc_success": 100.0 * sum(primary_successes) / len(primary_successes),
            "n_primary_rollouts": len(primary_successes),
        },
        "per_task": task_results,
    }


@parser.wrap()
def eval_smolvla_icl_main(cfg: SmolVLAICLEvalPipelineConfig) -> None:
    logging.info(pformat(asdict(cfg)))
    device = get_safe_torch_device(cfg.policy.device, log=True)
    set_seed(cfg.seed)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    manifest = LiberoDataManifest.load(cfg.manifest_path)
    train_stats = TrainStatsArtifact.load(cfg.train_stats_path, manifest=manifest)
    _validate_timing(cfg, manifest)
    state_normalizer = DemoStateNormalizer.from_dataset_stats(
        train_stats.to_dataset_stats(state_key=cfg.state_key),
        state_key=cfg.state_key,
    )
    demo_provider = TestDemoProvider(
        manifest,
        dataset_root=cfg.dataset_root,
        state_key=cfg.state_key,
        video_backend=cfg.video_backend,
        tolerance_s=cfg.tolerance_s,
    )
    matcher, matcher_snapshot = _make_matcher(cfg, device)

    envs = make_env(cfg.env, n_envs=1, use_async_envs=False, trust_remote_code=cfg.trust_remote_code)
    try:
        policy = make_policy(cfg=cfg.policy, env_cfg=cfg.env, rename_map=cfg.rename_map)
        policy.eval()
        processor_stats = train_stats.to_dataset_stats()
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=cfg.policy,
            pretrained_path=cfg.policy.pretrained_path,
            preprocessor_overrides={
                "device_processor": {"device": str(policy.config.device)},
                "rename_observations_processor": {"rename_map": cfg.rename_map},
                # 显式使用当前已验 fingerprint 的 train-only stats，避免
                # checkpoint 内残留的旧 processor state 静默污染 test。
                "normalizer_processor": {"stats": processor_stats},
            },
            postprocessor_overrides={
                "unnormalizer_processor": {"stats": processor_stats},
            },
        )
        env_preprocessor, env_postprocessor = make_env_pre_post_processors(
            env_cfg=cfg.env,
            policy_cfg=cfg.policy,
        )
        autocast = torch.autocast(device_type=device.type) if cfg.policy.use_amp else nullcontext()
        with torch.no_grad(), autocast:
            info = evaluate_smolvla_icl(
                cfg,
                envs=envs,
                policy=policy,
                env_preprocessor=env_preprocessor,
                env_postprocessor=env_postprocessor,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                manifest=manifest,
                demo_provider=demo_provider,
                state_normalizer=state_normalizer,
                matcher=matcher,
                matcher_snapshot=matcher_snapshot,
                train_stats_fingerprint=train_stats.fingerprint,
            )
    finally:
        close_envs(envs)

    output_path = cfg.output_dir / "eval_info.json"
    output_path.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    logger.info("SmolVLA-ICL online test complete: %s", output_path)
    logger.info("Overall: %s", info["overall"])


def main() -> None:
    init_logging()
    register_third_party_plugins()
    eval_smolvla_icl_main()


if __name__ == "__main__":
    main()
