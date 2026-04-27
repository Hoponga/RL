# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
"""Self-distillation summarizer training.

Single policy ``π_θ`` plays three roles in each step:
  - Teacher: scores a continuation conditioned on the *full* uncompressed prefix.
  - Summarizer: produces a short summary of the prefix.
  - Student: scores the same continuation conditioned on the summary.

Two losses are applied per step:
  - Part A (summarizer reward): REINFORCE on the summary tokens. Reward is a
    sample-level estimate of -KL(p_T || p_S) on a short continuation window:
        R_i = mean_t [ log p_S(y_t | summary, y_<t) - log p_T(y_t | prefix, y_<t) ]
    Since y_t was sampled from p_T(·|prefix), this is an unbiased per-position
    estimate of -KL(p_T || p_S).
  - Part B (context distillation): per-token KL on the continuation under the
    summarized prefix, using the existing DistillationLossFn with teacher
    top-k support computed from the full prefix.
"""
import os
import random
import warnings
from pathlib import Path
from typing import Any, NotRequired, Optional, TypedDict, TypeVar, cast

import numpy as np
import ray
import torch
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from nemo_rl.algorithms.grpo import _should_use_async_rollouts, refit_policy_generation
from nemo_rl.algorithms.loss import (
    DistillationLossConfig,
    DistillationLossDataDict,
    DistillationLossFn,
)
from nemo_rl.algorithms.loss.interfaces import (
    LossFunction,
    LossInputType,
    LossType,
)
from nemo_rl.algorithms.utils import masked_mean, set_seed
from nemo_rl.data import DataConfig
from nemo_rl.data.collate_fn import rl_collate_fn
from nemo_rl.data.datasets import AllTaskProcessedDataset
from nemo_rl.data.interfaces import DatumSpec
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.ray_actor_environment_registry import (
    ACTOR_ENVIRONMENT_REGISTRY,
)
from nemo_rl.distributed.virtual_cluster import (
    ClusterConfig,
    PY_EXECUTABLES,
    RayVirtualCluster,
)
from nemo_rl.environments.interfaces import EnvironmentInterface
from nemo_rl.experience.rollouts import (
    run_async_multi_turn_rollout,
    run_multi_turn_rollout,
)
from nemo_rl.models.generation.interfaces import GenerationInterface
from nemo_rl.models.generation.vllm import VllmConfig, VllmGeneration
from nemo_rl.models.policy import PolicyConfig
from nemo_rl.models.policy.interfaces import ColocatablePolicyInterface
from nemo_rl.models.policy.lm_policy import Policy
from nemo_rl.utils.checkpoint import CheckpointingConfig, CheckpointManager
from nemo_rl.utils.logger import (
    Logger,
    LoggerConfig,
    print_message_log_samples,
)
from nemo_rl.utils.nsys import maybe_gpu_profile_step
from nemo_rl.utils.timer import TimeoutChecker, Timer

# Reuse the same worker venv registration as self-distillation.
SUMMARIZER_WORKER_FQN = (
    "nemo_rl.algorithms.distillation_worker_extension.DistillationWorkerV2"
)
ACTOR_ENVIRONMENT_REGISTRY[SUMMARIZER_WORKER_FQN] = PY_EXECUTABLES.AUTOMODEL

TokenizerType = TypeVar("TokenizerType", bound=PreTrainedTokenizerBase)


# ===============================================================================
# Configuration
# ===============================================================================
class SummarizerDistillationConfig(TypedDict):
    num_prompts_per_step: int
    num_generations_per_prompt: int
    max_rollout_turns: int
    max_num_steps: int
    max_num_epochs: int
    val_batch_size: int
    val_period: int
    val_at_start: bool
    val_at_end: bool
    max_val_samples: int
    topk_logits_k: int
    seed: int

    # Summarizer-specific knobs.
    chunk_size: int  # tokens per reasoning chunk
    min_prefix_chunks: int  # smallest number of leading chunks to compress
    max_prefix_chunks: int  # largest number of leading chunks to compress
    continuation_window: int  # short-horizon length for KL/reward (tokens)
    summary_max_tokens: int
    summary_temperature: float
    summary_instruction: str  # prompt text used to elicit summaries
    reward_baseline: str  # "none" | "mean" | "mean_std"
    summarizer_loss_weight: float  # scale on Part A
    distillation_loss_weight: float  # scale on Part B


class AdvantageNLLLossConfig(TypedDict):
    pass


class SummarizerSaveState(TypedDict):
    total_steps: int
    current_epoch: int
    current_step: int
    val_reward: NotRequired[float]
    consumed_samples: int
    total_valid_tokens: int


def _default_save_state() -> SummarizerSaveState:
    return {
        "current_epoch": 0,
        "current_step": 0,
        "total_steps": 0,
        "val_reward": -99999999.0,
        "consumed_samples": 0,
        "total_valid_tokens": 0,
    }


class MasterConfig(TypedDict):
    policy: PolicyConfig
    loss_fn: DistillationLossConfig
    env: dict[str, Any]
    data: DataConfig
    distillation: SummarizerDistillationConfig
    logger: LoggerConfig
    cluster: ClusterConfig
    checkpointing: CheckpointingConfig


# ===============================================================================
# Custom loss: advantage-weighted NLL (REINFORCE on summary tokens)
# ===============================================================================
class AdvantageNLLLossFn(LossFunction):
    """REINFORCE-style loss: minimize ``-A * log p(token)`` on masked positions.

    Expects ``data["advantages"]`` of shape ``[B]`` (one scalar per sample).
    Uses ``token_mask`` to select tokens that belong to the summary segment.
    """

    loss_type = LossType.TOKEN_LEVEL
    input_type = LossInputType.LOGPROB

    def __init__(self, cfg: Optional[AdvantageNLLLossConfig] = None):
        del cfg

    def __call__(
        self,
        next_token_logprobs: torch.Tensor,
        data: BatchedDataDict[Any],
        global_valid_seqs: torch.Tensor | None,
        global_valid_toks: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        token_mask = data["token_mask"][:, 1:]
        sample_mask = data["sample_mask"]
        advantages = data["advantages"].to(next_token_logprobs.dtype)

        mask = token_mask * sample_mask.unsqueeze(-1)
        weighted = next_token_logprobs * advantages.unsqueeze(-1)
        loss = -masked_mean(
            weighted,
            mask,
            global_normalization_factor=global_valid_toks,
        )

        return loss, {
            "loss": loss.item() if loss.ndim == 0 else loss,
            "advantage_mean": advantages.mean().item(),
            "advantage_std": advantages.std(unbiased=False).item(),
            "num_unmasked_tokens": mask.sum().item(),
            "num_valid_samples": sample_mask.sum().item(),
        }


# ===============================================================================
# Rollout chunking + summary prompt construction
# ===============================================================================
def _extract_question_and_assistant(
    message_log: list[dict[str, Any]],
    tokenizer: TokenizerType,
) -> tuple[torch.Tensor, torch.Tensor, str]:
    """Return ``(prompt_token_ids, assistant_token_ids, assistant_text)``.

    ``prompt_token_ids`` is the concatenation of every non-assistant message's
    tokens (in order, with chat template tokens preserved).
    """
    prompt_chunks: list[torch.Tensor] = []
    assistant_tokens: Optional[torch.Tensor] = None
    assistant_text = ""

    for msg in message_log:
        token_ids = msg.get("token_ids")
        if token_ids is None:
            continue
        if msg["role"] == "assistant" and assistant_tokens is None:
            assistant_tokens = token_ids
            assistant_text = msg.get("content", "") or ""
        else:
            prompt_chunks.append(token_ids)

    if not prompt_chunks:
        prompt_ids = torch.tensor([], dtype=torch.long)
    else:
        prompt_ids = torch.cat(prompt_chunks, dim=0).to(torch.long)

    if assistant_tokens is None:
        assistant_tokens = torch.tensor([], dtype=torch.long)
    else:
        assistant_tokens = assistant_tokens.to(torch.long)

    return prompt_ids, assistant_tokens, assistant_text


def _chunk_assistant(
    assistant_token_ids: torch.Tensor,
    chunk_size: int,
) -> list[torch.Tensor]:
    if assistant_token_ids.numel() == 0:
        return []
    return [
        assistant_token_ids[i : i + chunk_size]
        for i in range(0, assistant_token_ids.numel(), chunk_size)
    ]


def _pick_split_index(
    num_chunks: int,
    min_prefix_chunks: int,
    max_prefix_chunks: int,
    rng: random.Random,
) -> int:
    """Pick ``k`` such that prefix = chunks[:k] and continuation = chunks[k:]."""
    lo = max(1, min_prefix_chunks)
    hi = min(num_chunks - 1, max_prefix_chunks)
    if hi < lo:
        return -1  # not enough chunks for a valid split
    return rng.randint(lo, hi)


def _build_summarize_prompt_ids(
    tokenizer: TokenizerType,
    prompt_ids: torch.Tensor,
    prefix_chunks: list[torch.Tensor],
    summary_instruction: str,
) -> torch.Tensor:
    """Build the input passed to vLLM to elicit a summary.

    Layout:
        [question chat-template tokens] [prefix tokens] [summary_instruction tokens]
    The summary_instruction is appended as raw tokens (no chat template wrapper),
    so it should already include any phrasing that signals the model to produce
    a concise summary on the next turn.
    """
    pieces: list[torch.Tensor] = [prompt_ids.to(torch.long)]
    pieces.extend(c.to(torch.long) for c in prefix_chunks)
    instr_ids = tokenizer(
        summary_instruction, add_special_tokens=False, return_tensors="pt"
    )["input_ids"][0].to(torch.long)
    pieces.append(instr_ids)
    return torch.cat(pieces, dim=0)


def _right_pad(
    sequences: list[torch.Tensor],
    pad_token_id: int,
    multiple_of: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not sequences:
        return (
            torch.zeros((0, 0), dtype=torch.long),
            torch.zeros((0,), dtype=torch.long),
        )
    lengths = torch.tensor([s.numel() for s in sequences], dtype=torch.long)
    max_len = int(lengths.max().item())
    if multiple_of > 1 and max_len % multiple_of != 0:
        max_len = ((max_len // multiple_of) + 1) * multiple_of
    out = torch.full((len(sequences), max_len), pad_token_id, dtype=torch.long)
    for i, s in enumerate(sequences):
        out[i, : s.numel()] = s.to(torch.long)
    return out, lengths


# ===============================================================================
# Per-step example construction
# ===============================================================================
class _Example(TypedDict):
    """Aligned per-sample tensors used by every loss in the step."""

    prompt_ids: torch.Tensor          # question + chat-template prefix (1D long)
    full_prefix_ids: torch.Tensor     # prefix chunks concatenated (1D long)
    continuation_ids: torch.Tensor    # short window (1D long)
    summary_input_ids: torch.Tensor   # what we passed to vLLM to elicit summary (1D)
    summary_ids: torch.Tensor         # generated summary tokens (1D long)
    valid: bool                       # whether this sample participates in losses


def _build_examples(
    repeated_batch: BatchedDataDict[DatumSpec],
    tokenizer: TokenizerType,
    chunk_size: int,
    min_prefix_chunks: int,
    max_prefix_chunks: int,
    continuation_window: int,
    summary_instruction: str,
    rng: random.Random,
) -> list[_Example]:
    examples: list[_Example] = []
    message_logs = repeated_batch["message_log"]
    for ml in message_logs:
        prompt_ids, assistant_ids, _assistant_text = _extract_question_and_assistant(
            ml, tokenizer
        )
        chunks = _chunk_assistant(assistant_ids, chunk_size)
        if len(chunks) < max(2, min_prefix_chunks + 1):
            examples.append(
                _Example(
                    prompt_ids=prompt_ids,
                    full_prefix_ids=torch.tensor([], dtype=torch.long),
                    continuation_ids=torch.tensor([], dtype=torch.long),
                    summary_input_ids=torch.tensor([], dtype=torch.long),
                    summary_ids=torch.tensor([], dtype=torch.long),
                    valid=False,
                )
            )
            continue

        k = _pick_split_index(
            len(chunks), min_prefix_chunks, max_prefix_chunks, rng
        )
        if k < 0:
            examples.append(
                _Example(
                    prompt_ids=prompt_ids,
                    full_prefix_ids=torch.tensor([], dtype=torch.long),
                    continuation_ids=torch.tensor([], dtype=torch.long),
                    summary_input_ids=torch.tensor([], dtype=torch.long),
                    summary_ids=torch.tensor([], dtype=torch.long),
                    valid=False,
                )
            )
            continue

        prefix_chunks = chunks[:k]
        full_prefix_ids = torch.cat(prefix_chunks, dim=0)
        # Continuation = first `continuation_window` tokens after the prefix.
        tail = torch.cat(chunks[k:], dim=0)
        continuation_ids = tail[:continuation_window]

        summary_input_ids = _build_summarize_prompt_ids(
            tokenizer, prompt_ids, prefix_chunks, summary_instruction
        )

        examples.append(
            _Example(
                prompt_ids=prompt_ids,
                full_prefix_ids=full_prefix_ids,
                continuation_ids=continuation_ids,
                summary_input_ids=summary_input_ids,
                summary_ids=torch.tensor([], dtype=torch.long),
                valid=continuation_ids.numel() > 0,
            )
        )
    return examples


# ===============================================================================
# Helpers for assembling teacher / student / summarizer training tensors
# ===============================================================================
def _continuation_mask(
    seq_len: int, cont_start: int, cont_end: int, device: torch.device
) -> torch.Tensor:
    mask = torch.zeros(seq_len, dtype=torch.long, device=device)
    if cont_end > cont_start:
        mask[cont_start:cont_end] = 1
    return mask


def _scatter_topk(
    teacher_topk: BatchedDataDict[Any],
    teacher_input_lengths: torch.Tensor,
    teacher_cont_starts: list[int],
    student_seq_len: int,
    student_cont_starts: list[int],
    cont_lens: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Move teacher's continuation top-k entries into the student frame.

    Teacher and student share the same continuation tokens but live at different
    sequence positions because their prefixes have different lengths. We copy
    the teacher's [cont_start, cont_end) slice into the student's slice.
    """
    topk_logits = teacher_topk["topk_logits"]
    topk_indices = teacher_topk["topk_indices"]
    batch_size = topk_logits.shape[0]
    k = topk_logits.shape[-1]

    aligned_logits = torch.zeros(
        (batch_size, student_seq_len, k),
        dtype=topk_logits.dtype,
        device=topk_logits.device,
    )
    aligned_indices = torch.zeros(
        (batch_size, student_seq_len, k),
        dtype=topk_indices.dtype,
        device=topk_indices.device,
    )

    for i in range(batch_size):
        cont_len = cont_lens[i]
        if cont_len <= 0:
            continue
        t_start = teacher_cont_starts[i]
        s_start = student_cont_starts[i]
        t_end = min(t_start + cont_len, int(teacher_input_lengths[i].item()))
        actual = max(0, t_end - t_start)
        if actual <= 0:
            continue
        aligned_logits[i, s_start : s_start + actual] = topk_logits[
            i, t_start : t_start + actual
        ]
        aligned_indices[i, s_start : s_start + actual] = topk_indices[
            i, t_start : t_start + actual
        ]
    return aligned_logits, aligned_indices


def _gather_logprobs_at_continuation(
    logprobs: torch.Tensor,
    cont_starts: list[int],
    cont_lens: list[int],
) -> list[torch.Tensor]:
    """Slice each row's per-token logprobs over the continuation window.

    ``logprobs[i, j]`` is the logprob of ``input_ids[i, j]`` given prior tokens
    (with ``logprobs[i, 0] == 0`` per the policy convention).
    """
    out: list[torch.Tensor] = []
    for i, (s, n) in enumerate(zip(cont_starts, cont_lens)):
        if n <= 0:
            out.append(torch.zeros((0,), dtype=logprobs.dtype))
            continue
        out.append(logprobs[i, s : s + n].detach().cpu())
    return out


# ===============================================================================
# Setup
# ===============================================================================
def setup(
    master_config: MasterConfig,
    tokenizer: TokenizerType,
    train_dataset: AllTaskProcessedDataset,
    val_dataset: Optional[AllTaskProcessedDataset],
) -> tuple[
    ColocatablePolicyInterface,
    Optional[GenerationInterface],
    StatefulDataLoader,
    Optional[StatefulDataLoader],
    DistillationLossFn,
    AdvantageNLLLossFn,
    Logger,
    CheckpointManager,
    SummarizerSaveState,
    MasterConfig,
]:
    policy_config = master_config["policy"]
    generation_config = master_config["policy"]["generation"]
    loss_config = master_config["loss_fn"]
    distillation_config = master_config["distillation"]
    data_config = master_config["data"]
    logger_config = master_config["logger"]
    cluster_config = master_config["cluster"]

    assert generation_config is not None, (
        "policy.generation is required (we need to sample teacher rollouts and summaries)"
    )

    set_seed(distillation_config["seed"])

    logger = Logger(logger_config)
    logger.log_hyperparams(master_config)

    checkpointer = CheckpointManager(master_config["checkpointing"])
    last_checkpoint_path = checkpointer.get_latest_checkpoint_path()
    save_state: Optional[SummarizerSaveState] = cast(
        Optional[SummarizerSaveState],
        checkpointer.load_training_info(last_checkpoint_path),
    )
    if save_state is None:
        save_state = _default_save_state()

    dataloader = StatefulDataLoader(
        train_dataset,
        batch_size=distillation_config["num_prompts_per_step"],
        shuffle=data_config["shuffle"],
        collate_fn=rl_collate_fn,
        drop_last=True,
    )
    if last_checkpoint_path:
        dataloader.load_state_dict(
            torch.load(os.path.join(last_checkpoint_path, "train_dataloader.pt"))
        )

    val_dataloader: Optional[StatefulDataLoader] = None
    if (
        distillation_config["val_period"] > 0
        or distillation_config["val_at_start"]
        or distillation_config["val_at_end"]
    ):
        assert val_dataset is not None
        val_dataloader = StatefulDataLoader(
            val_dataset,
            batch_size=distillation_config["val_batch_size"],
            shuffle=False,
            collate_fn=rl_collate_fn,
        )

    print("\n▶ Setting up compute cluster...", flush=True)
    colocated_inference = generation_config["colocated"]["enabled"]
    assert colocated_inference, (
        "self_distillation_summarizer currently assumes colocated generation."
    )
    cluster = RayVirtualCluster(
        name="summarizer_cluster",
        bundle_ct_per_node_list=[cluster_config["gpus_per_node"]]
        * cluster_config["num_nodes"],
        use_gpus=True,
        num_gpus_per_node=cluster_config["gpus_per_node"],
        max_colocated_worker_groups=1
        if generation_config["backend"] == "megatron"
        else 3,
    )

    backend = generation_config["backend"]
    generation_config["model_name"] = policy_config["model_name"]
    if backend == "megatron":
        generation: Optional[GenerationInterface] = None
    elif backend == "vllm":
        gen_cfg = cast(VllmConfig, generation_config)
        if "vllm_cfg" in gen_cfg:
            gen_cfg["vllm_cfg"]["hf_overrides"] = policy_config.get(
                "hf_config_overrides", {}
            )
        generation = VllmGeneration(cluster=cluster, config=gen_cfg)
        generation.finish_generation()
    else:
        raise ValueError(f"Unsupported generation backend: {backend}")

    if last_checkpoint_path:
        weights_path: Optional[Path] = (
            Path(last_checkpoint_path) / "policy" / "weights"
        )
        optimizer_path: Optional[Path] = (
            Path(last_checkpoint_path) / "policy" / "optimizer"
        )
    else:
        weights_path = None
        optimizer_path = None

    if "megatron_cfg" in policy_config and policy_config["megatron_cfg"]["enabled"]:
        total_train_iters = min(
            distillation_config["max_num_steps"],
            distillation_config["max_num_epochs"] * len(dataloader),
        )
        policy_config["megatron_cfg"]["train_iters"] = total_train_iters

    policy = Policy(
        name_prefix="summarizer",
        cluster=cluster,
        config=policy_config,
        tokenizer=tokenizer,
        weights_path=weights_path,
        optimizer_path=optimizer_path,
        init_optimizer=True,
        init_reference_model=False,
        worker_extension_cls_fqn=SUMMARIZER_WORKER_FQN,
    )

    if generation is not None:
        state_dict_info = policy.prepare_refit_info()
        generation.prepare_refit_info(state_dict_info)

    distill_loss_fn = DistillationLossFn(loss_config)
    summarizer_loss_fn = AdvantageNLLLossFn()

    return (
        policy,
        generation,
        dataloader,
        val_dataloader,
        distill_loss_fn,
        summarizer_loss_fn,
        logger,
        checkpointer,
        save_state,
        master_config,
    )


# ===============================================================================
# Training step
# ===============================================================================
def _generate_summaries(
    generation: GenerationInterface,
    examples: list[_Example],
    tokenizer: TokenizerType,
    summary_max_tokens: int,
    summary_temperature: float,
    pad_multiple_of: int,
) -> list[torch.Tensor]:
    """Run one vLLM generation pass to summarize every valid example."""
    valid_idx = [i for i, ex in enumerate(examples) if ex["valid"]]
    if not valid_idx:
        return [torch.tensor([], dtype=torch.long) for _ in examples]

    sequences = [examples[i]["summary_input_ids"] for i in valid_idx]
    pad_id = tokenizer.pad_token_id or 0
    input_ids, input_lengths = _right_pad(sequences, pad_id, multiple_of=pad_multiple_of)

    gen_data = BatchedDataDict({
        "input_ids": input_ids,
        "input_lengths": input_lengths,
    })

    out = generation.generate(gen_data, greedy=False)
    output_ids = out["output_ids"]
    gen_lengths = out["generation_lengths"]

    summaries = [torch.tensor([], dtype=torch.long) for _ in examples]
    for slot, src in enumerate(valid_idx):
        in_len = int(input_lengths[slot].item())
        out_len = int(gen_lengths[slot].item())
        out_len = min(out_len, summary_max_tokens)
        summary_tokens = output_ids[slot, in_len : in_len + out_len].clone().to(torch.long)
        summaries[src] = summary_tokens
    return summaries


def _build_teacher_student_seqs(
    examples: list[_Example],
    pad_token_id: int,
    pad_multiple_of: int,
) -> dict[str, Any]:
    """Construct teacher_seq, student_seq, and continuation alignment metadata."""
    teacher_seqs: list[torch.Tensor] = []
    student_seqs: list[torch.Tensor] = []
    teacher_cont_starts: list[int] = []
    student_cont_starts: list[int] = []
    cont_lens: list[int] = []
    valid_flags: list[float] = []

    for ex in examples:
        if not ex["valid"]:
            teacher_seqs.append(torch.tensor([0], dtype=torch.long))
            student_seqs.append(torch.tensor([0], dtype=torch.long))
            teacher_cont_starts.append(0)
            student_cont_starts.append(0)
            cont_lens.append(0)
            valid_flags.append(0.0)
            continue

        prompt = ex["prompt_ids"]
        full_prefix = ex["full_prefix_ids"]
        summary = ex["summary_ids"]
        cont = ex["continuation_ids"]

        teacher_seq = torch.cat([prompt, full_prefix, cont], dim=0)
        student_seq = torch.cat([prompt, summary, cont], dim=0)

        teacher_cont_starts.append(prompt.numel() + full_prefix.numel())
        student_cont_starts.append(prompt.numel() + summary.numel())
        cont_lens.append(cont.numel())
        teacher_seqs.append(teacher_seq)
        student_seqs.append(student_seq)
        valid_flags.append(1.0 if cont.numel() > 0 and summary.numel() > 0 else 0.0)

    teacher_ids, teacher_lens = _right_pad(teacher_seqs, pad_token_id, pad_multiple_of)
    student_ids, student_lens = _right_pad(student_seqs, pad_token_id, pad_multiple_of)

    return {
        "teacher_input_ids": teacher_ids,
        "teacher_input_lengths": teacher_lens,
        "student_input_ids": student_ids,
        "student_input_lengths": student_lens,
        "teacher_cont_starts": teacher_cont_starts,
        "student_cont_starts": student_cont_starts,
        "cont_lens": cont_lens,
        "valid_flags": torch.tensor(valid_flags, dtype=torch.float32),
    }


def _build_summarizer_train_data(
    examples: list[_Example],
    advantages: torch.Tensor,
    valid_flags: torch.Tensor,
    pad_token_id: int,
    pad_multiple_of: int,
) -> Optional[BatchedDataDict[Any]]:
    """Build inputs for the REINFORCE update on summary tokens."""
    sequences: list[torch.Tensor] = []
    summary_starts: list[int] = []
    summary_lens: list[int] = []
    keep: list[int] = []

    for i, ex in enumerate(examples):
        if not ex["valid"] or ex["summary_ids"].numel() == 0:
            continue
        full = torch.cat([ex["summary_input_ids"], ex["summary_ids"]], dim=0)
        sequences.append(full)
        summary_starts.append(ex["summary_input_ids"].numel())
        summary_lens.append(ex["summary_ids"].numel())
        keep.append(i)

    if not sequences:
        return None

    input_ids, input_lengths = _right_pad(sequences, pad_token_id, pad_multiple_of)
    seq_len = input_ids.shape[1]
    token_mask = torch.zeros((len(sequences), seq_len), dtype=torch.long)
    for row, (s, n) in enumerate(zip(summary_starts, summary_lens)):
        token_mask[row, s : s + n] = 1

    sample_mask = valid_flags[keep].clone()
    selected_advantages = advantages[keep].clone()

    train_data = BatchedDataDict({
        "input_ids": input_ids,
        "input_lengths": input_lengths,
        "token_mask": token_mask,
        "sample_mask": sample_mask,
        "advantages": selected_advantages,
    })
    return train_data


def _normalize_rewards(rewards: torch.Tensor, mode: str) -> torch.Tensor:
    if rewards.numel() == 0:
        return rewards
    if mode == "none":
        return rewards
    centered = rewards - rewards.mean()
    if mode == "mean":
        return centered
    if mode == "mean_std":
        std = rewards.std(unbiased=False)
        return centered / (std + 1e-6)
    raise ValueError(f"Unknown reward_baseline mode: {mode}")


def summarizer_train(
    policy: ColocatablePolicyInterface,
    generation: Optional[GenerationInterface],
    dataloader: StatefulDataLoader,
    val_dataloader: Optional[StatefulDataLoader],
    tokenizer: TokenizerType,
    distill_loss_fn: DistillationLossFn,
    summarizer_loss_fn: AdvantageNLLLossFn,
    task_to_env: dict[str, EnvironmentInterface],
    val_task_to_env: Optional[dict[str, EnvironmentInterface]],
    logger: Logger,
    checkpointer: CheckpointManager,
    save_state: SummarizerSaveState,
    master_config: MasterConfig,
) -> None:
    timer = Timer()
    timeout = TimeoutChecker(
        timeout=master_config["checkpointing"]["checkpoint_must_save_by"],
        fit_last_save_time=True,
    )
    timeout.start_iterations()

    NEED_REFIT = True
    if generation is None:
        generation = policy  # megatron path: same actor handles both
        NEED_REFIT = False
    POLICY_GENERATION_STALE = True
    assert generation is not None

    distill_cfg = master_config["distillation"]
    current_epoch = save_state["current_epoch"]
    current_step = save_state["current_step"]
    total_steps = save_state["total_steps"]
    consumed_samples = save_state["consumed_samples"]
    total_valid_tokens = save_state["total_valid_tokens"]
    val_period = distill_cfg["val_period"]
    val_at_start = distill_cfg["val_at_start"]
    val_at_end = distill_cfg["val_at_end"]
    colocated_inference = master_config["policy"]["generation"]["colocated"]["enabled"]
    max_epochs = distill_cfg["max_num_epochs"]
    max_steps = distill_cfg["max_num_steps"]
    pad_token_id = tokenizer.pad_token_id or 0
    pad_multiple_of = master_config["policy"].get(
        "make_sequence_length_divisible_by", 1
    )
    rng = random.Random(distill_cfg["seed"] + 0xBEEF)

    if val_at_start and total_steps == 0 and val_dataloader is not None:
        if NEED_REFIT and POLICY_GENERATION_STALE:
            refit_policy_generation(policy, generation, colocated_inference)
            POLICY_GENERATION_STALE = False
        else:
            generation.prepare_for_generation()
        val_metrics, validation_timings = validate(
            generation,
            val_dataloader,
            tokenizer,
            val_task_to_env,
            step=total_steps,
            master_config=master_config,
        )
        generation.finish_generation()
        logger.log_metrics(val_metrics, total_steps, prefix="validation")
        logger.log_metrics(validation_timings, total_steps, prefix="timing/validation")

    batch: BatchedDataDict[DatumSpec]
    while total_steps < max_steps and current_epoch < max_epochs:
        print(
            f"\n{'=' * 25} Epoch {current_epoch + 1}/{max_epochs} {'=' * 25}",
            flush=True,
        )
        for batch in dataloader:
            print(
                f"\n{'=' * 25} Step {current_step + 1}/{min(len(dataloader), max_steps)} {'=' * 25}",
                flush=True,
            )
            maybe_gpu_profile_step(policy, total_steps + 1)
            if policy != generation:
                maybe_gpu_profile_step(generation, total_steps + 1)
            val_metrics, validation_timings = None, None

            with timer.time("total_step_time"):
                # ---- 1. Teacher rollout: have the model solve the problem.
                with timer.time("data_processing"):
                    repeated_batch = batch.repeat_interleave(
                        distill_cfg["num_generations_per_prompt"]
                    )

                with timer.time("prepare_for_generation"):
                    if NEED_REFIT and POLICY_GENERATION_STALE:
                        refit_policy_generation(
                            policy, generation, colocated_inference, timer=timer
                        )
                        POLICY_GENERATION_STALE = False
                    else:
                        generation.prepare_for_generation()

                with timer.time("generation"):
                    if _should_use_async_rollouts(master_config):
                        repeated_batch, rollout_metrics = run_async_multi_turn_rollout(
                            policy_generation=generation,
                            input_batch=repeated_batch,
                            tokenizer=tokenizer,
                            task_to_env=task_to_env,
                            max_seq_len=master_config["policy"][
                                "max_total_sequence_length"
                            ],
                            max_rollout_turns=distill_cfg["max_rollout_turns"],
                            greedy=False,
                        )
                    else:
                        repeated_batch, rollout_metrics = run_multi_turn_rollout(
                            policy_generation=generation,
                            input_batch=repeated_batch,
                            tokenizer=tokenizer,
                            task_to_env=task_to_env,
                            max_seq_len=master_config["policy"][
                                "max_total_sequence_length"
                            ],
                            max_rollout_turns=distill_cfg["max_rollout_turns"],
                            greedy=False,
                        )

                # ---- 2. Chunk + 3. pick split + prepare summary inputs.
                examples = _build_examples(
                    repeated_batch,
                    tokenizer,
                    chunk_size=distill_cfg["chunk_size"],
                    min_prefix_chunks=distill_cfg["min_prefix_chunks"],
                    max_prefix_chunks=distill_cfg["max_prefix_chunks"],
                    continuation_window=distill_cfg["continuation_window"],
                    summary_instruction=distill_cfg["summary_instruction"],
                    rng=rng,
                )

                # ---- 4. Summarize.
                with timer.time("summary_generation"):
                    summary_token_lists = _generate_summaries(
                        generation,
                        examples,
                        tokenizer,
                        summary_max_tokens=distill_cfg["summary_max_tokens"],
                        summary_temperature=distill_cfg["summary_temperature"],
                        pad_multiple_of=pad_multiple_of,
                    )
                generation.finish_generation()

                for ex, sm in zip(examples, summary_token_lists):
                    ex["summary_ids"] = sm
                    if ex["valid"] and sm.numel() == 0:
                        ex["valid"] = False

                # ---- 5. Build aligned teacher / student sequences.
                seqs = _build_teacher_student_seqs(
                    examples, pad_token_id, pad_multiple_of
                )
                valid_flags = seqs["valid_flags"]
                num_valid = int(valid_flags.sum().item())
                if num_valid == 0:
                    print(
                        "  ⚠️  No valid summarizer examples this step (rollouts too short). Skipping.",
                        flush=True,
                    )
                    current_step += 1
                    total_steps += 1
                    continue

                # ---- 6. Get per-token logprobs for reward (no grad).
                with timer.time("logprob_inference_prep"):
                    policy.prepare_for_lp_inference()

                teacher_data = BatchedDataDict({
                    "input_ids": seqs["teacher_input_ids"],
                    "input_lengths": seqs["teacher_input_lengths"],
                })
                student_data = BatchedDataDict({
                    "input_ids": seqs["student_input_ids"],
                    "input_lengths": seqs["student_input_lengths"],
                })

                with timer.time("teacher_logprobs"):
                    teacher_logprobs = policy.get_logprobs(teacher_data, timer=timer)[
                        "logprobs"
                    ]
                with timer.time("student_logprobs"):
                    student_logprobs = policy.get_logprobs(student_data, timer=timer)[
                        "logprobs"
                    ]

                # ---- 6b. Teacher top-k for the differentiable distillation loss.
                with timer.time("teacher_topk"):
                    teacher_topk = policy.get_topk_logits(
                        teacher_data,
                        k=distill_cfg["topk_logits_k"],
                        timer=timer,
                    )

                # Per-sample reward = mean_t [log p_S(y_t) - log p_T(y_t)] over the
                # continuation, using y_t sampled from p_T(·|prefix) so this is an
                # unbiased estimate of -KL(p_T || p_S).
                rewards = torch.zeros(len(examples), dtype=torch.float32)
                for i in range(len(examples)):
                    n = seqs["cont_lens"][i]
                    if n <= 0 or not bool(valid_flags[i].item()):
                        continue
                    s_t = seqs["teacher_cont_starts"][i]
                    s_s = seqs["student_cont_starts"][i]
                    t_lp = teacher_logprobs[i, s_t : s_t + n].to(torch.float32)
                    s_lp = student_logprobs[i, s_s : s_s + n].to(torch.float32)
                    if t_lp.numel() == 0 or s_lp.numel() == 0:
                        continue
                    rewards[i] = (s_lp.mean() - t_lp.mean()).item()

                advantages = _normalize_rewards(
                    rewards, mode=distill_cfg.get("reward_baseline", "mean")
                )

                # ---- 7. Distillation training data (Part B).
                student_seq_len = seqs["student_input_ids"].shape[1]
                aligned_logits, aligned_indices = _scatter_topk(
                    teacher_topk,
                    seqs["teacher_input_lengths"],
                    seqs["teacher_cont_starts"],
                    student_seq_len,
                    seqs["student_cont_starts"],
                    seqs["cont_lens"],
                )
                if aligned_logits.device.type != "cpu":
                    aligned_logits = aligned_logits.cpu()
                    aligned_indices = aligned_indices.cpu()

                student_token_mask = torch.zeros(
                    (len(examples), student_seq_len), dtype=torch.long
                )
                for i in range(len(examples)):
                    n = seqs["cont_lens"][i]
                    if n > 0 and bool(valid_flags[i].item()):
                        s_s = seqs["student_cont_starts"][i]
                        student_token_mask[i, s_s : s_s + n] = 1

                distill_train_data = BatchedDataDict[DistillationLossDataDict]({
                    "input_ids": seqs["student_input_ids"],
                    "input_lengths": seqs["student_input_lengths"],
                    "token_mask": student_token_mask,
                    "sample_mask": valid_flags,
                    "teacher_topk_logits": aligned_logits,
                    "teacher_topk_indices": aligned_indices,
                })
                distill_train_data.to("cpu")

                # ---- 8. Summarizer training data (Part A).
                summarizer_train_data = _build_summarizer_train_data(
                    examples,
                    advantages,
                    valid_flags,
                    pad_token_id,
                    pad_multiple_of,
                )

                # ---- 9. Train.
                with timer.time("training_prep"):
                    policy.prepare_for_training()
                    POLICY_GENERATION_STALE = True

                summarizer_train_results: dict[str, Any] = {}
                if summarizer_train_data is not None:
                    print("▶ Training summarizer (REINFORCE on summary tokens)...", flush=True)
                    with timer.time("summarizer_training"):
                        summarizer_train_data.to("cpu")
                        summarizer_train_results = policy.train(
                            summarizer_train_data,
                            summarizer_loss_fn,
                            timer=timer,
                        )

                print("▶ Training distillation (KL on continuation)...", flush=True)
                with timer.time("distillation_training"):
                    distill_train_results = policy.train(
                        distill_train_data,
                        distill_loss_fn,
                        timer=timer,
                    )

                is_last_step = (total_steps + 1 >= max_steps) or (
                    (current_epoch + 1 == max_epochs)
                    and (current_step + 1 == len(dataloader))
                )

                if (val_period > 0 and (total_steps + 1) % val_period == 0) or (
                    val_at_end and is_last_step
                ):
                    if NEED_REFIT and POLICY_GENERATION_STALE:
                        refit_policy_generation(policy, generation, colocated_inference)
                        POLICY_GENERATION_STALE = False
                    else:
                        generation.prepare_for_generation()
                    val_metrics, validation_timings = validate(
                        generation,
                        val_dataloader,
                        tokenizer,
                        val_task_to_env,
                        step=total_steps + 1,
                        master_config=master_config,
                    )
                    generation.finish_generation()
                    logger.log_metrics(
                        validation_timings, total_steps + 1, prefix="timing/validation"
                    )
                    logger.log_metrics(val_metrics, total_steps + 1, prefix="validation")

                metrics: dict[str, Any] = {
                    "distill_loss": distill_train_results["loss"].numpy(),
                    "distill_grad_norm": distill_train_results["grad_norm"].numpy(),
                    "reward_mean": rewards[valid_flags.bool()].mean().item()
                    if num_valid > 0
                    else 0.0,
                    "reward_std": rewards[valid_flags.bool()].std(unbiased=False).item()
                    if num_valid > 1
                    else 0.0,
                    "num_valid_examples": float(num_valid),
                    "num_summary_tokens": float(
                        sum(ex["summary_ids"].numel() for ex in examples)
                    ),
                    "num_continuation_tokens": float(sum(seqs["cont_lens"])),
                }
                if summarizer_train_results:
                    metrics["summarizer_loss"] = summarizer_train_results["loss"].numpy()
                    metrics["summarizer_grad_norm"] = summarizer_train_results[
                        "grad_norm"
                    ].numpy()
                    metrics.update({
                        f"summarizer_{k}": v
                        for k, v in summarizer_train_results["all_mb_metrics"].items()
                    })
                metrics.update({
                    f"distill_{k}": v
                    for k, v in distill_train_results["all_mb_metrics"].items()
                })

                for k, v in list(metrics.items()):
                    if isinstance(v, np.ndarray):
                        metrics[k] = float(np.mean(v).item())
                    elif isinstance(v, list):
                        metrics[k] = float(np.sum(v).item()) if v else 0.0

                metrics.update(rollout_metrics)
                total_valid_tokens += int(metrics.get("num_continuation_tokens", 0))

                consumed_samples += distill_cfg["num_prompts_per_step"]
                timeout.mark_iteration()

                should_save_by_step = (
                    is_last_step
                    or (total_steps + 1) % master_config["checkpointing"]["save_period"]
                    == 0
                )
                should_save_by_timeout = timeout.check_save()

                if master_config["checkpointing"]["enabled"] and (
                    should_save_by_step or should_save_by_timeout
                ):
                    policy.prepare_for_training()

                    save_state["current_epoch"] = current_epoch
                    save_state["current_step"] = current_step + 1
                    save_state["total_steps"] = total_steps + 1
                    save_state["total_valid_tokens"] = total_valid_tokens
                    if val_metrics is not None:
                        save_state["val_reward"] = val_metrics["accuracy"]
                    elif "val_reward" in save_state:
                        del save_state["val_reward"]
                    save_state["consumed_samples"] = consumed_samples

                    full_metric_name = master_config["checkpointing"]["metric_name"]
                    if full_metric_name is not None:
                        assert full_metric_name.startswith(
                            "train:"
                        ) or full_metric_name.startswith("val:"), (
                            f"metric_name={full_metric_name} must start with 'val:' or 'train:'"
                        )
                        prefix, metric_name = full_metric_name.split(":", 1)
                        metrics_source = metrics if prefix == "train" else val_metrics
                        if not metrics_source:
                            warnings.warn(
                                f"You asked to save checkpoints based on {metric_name} but no {prefix} metrics were collected.",
                                stacklevel=2,
                            )
                            if full_metric_name in save_state:
                                del save_state[full_metric_name]
                        elif metric_name not in metrics_source:
                            raise ValueError(
                                f"Metric {metric_name} not found in {prefix} metrics"
                            )
                        else:
                            save_state[full_metric_name] = metrics_source[metric_name]

                    with timer.time("checkpointing"):
                        checkpoint_path = checkpointer.init_tmp_checkpoint(
                            total_steps + 1, save_state, master_config
                        )
                        policy.save_checkpoint(
                            weights_path=os.path.join(
                                checkpoint_path, "policy", "weights"
                            ),
                            optimizer_path=os.path.join(
                                checkpoint_path, "policy", "optimizer"
                            ),
                            tokenizer_path=os.path.join(
                                checkpoint_path, "policy", "tokenizer"
                            ),
                            checkpointing_cfg=master_config["checkpointing"],
                        )
                        torch.save(
                            dataloader.state_dict(),
                            os.path.join(checkpoint_path, "train_dataloader.pt"),
                        )
                        checkpointer.finalize_checkpoint(checkpoint_path)

            timing_metrics: dict[str, float] = timer.get_timing_metrics(
                reduction_op="sum"
            )  # type: ignore
            total_time = timing_metrics.get("total_step_time", 0)
            total_num_gpus = (
                master_config["cluster"]["num_nodes"]
                * master_config["cluster"]["gpus_per_node"]
            )
            metrics["tokens_per_sec_per_gpu"] = (
                metrics.get("num_continuation_tokens", 0)
                / total_time
                / max(1, total_num_gpus)
            )

            print("\n📊 Step Results:")
            print(f"  • distill_loss: {metrics.get('distill_loss', 0):.4f}")
            if "summarizer_loss" in metrics:
                print(f"  • summarizer_loss: {metrics['summarizer_loss']:.4f}")
            print(f"  • reward_mean: {metrics['reward_mean']:.4f}")
            print(f"  • reward_std:  {metrics['reward_std']:.4f}")
            print(f"  • valid examples: {int(metrics['num_valid_examples'])}")
            print(f"  • total step time: {total_time:.2f}s", flush=True)

            logger.log_metrics(metrics, total_steps + 1, prefix="train")
            logger.log_metrics(timing_metrics, total_steps + 1, prefix="timing/train")

            timer.reset()
            current_step += 1
            total_steps += 1
            if should_save_by_timeout:
                print("Timeout reached, stopping early", flush=True)
                return
            if total_steps >= max_steps:
                print("Max steps reached", flush=True)
                return

        current_epoch += 1
        current_step = 0


def validate(
    policy_generation: GenerationInterface,
    val_dataloader: Optional[StatefulDataLoader],
    tokenizer: TokenizerType,
    val_task_to_env: Optional[dict[str, EnvironmentInterface]],
    step: int,
    master_config: MasterConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validation reuses the standard rollout-based accuracy on the eval task.

    Summarizer quality is measured indirectly: a well-trained summarizer should
    not hurt downstream solve rate when its summaries are actually used at test
    time, but we don't insert summaries during validation here. Treat this as a
    regression check on the underlying policy.
    """
    if val_dataloader is None or val_task_to_env is None:
        return {}, {}

    timer = Timer()
    with timer.time("total_validation_time"):
        total_rewards: list[float] = []
        total_lengths: list[float] = []
        all_message_logs: list[Any] = []
        max_batches = (
            master_config["distillation"]["max_val_samples"]
            // master_config["distillation"]["val_batch_size"]
        )
        for batch_idx, val_batch in enumerate(val_dataloader):
            if batch_idx >= max_batches:
                break
            if _should_use_async_rollouts(master_config):
                val_batch, gen_metrics = run_async_multi_turn_rollout(
                    policy_generation,
                    val_batch,
                    tokenizer,
                    val_task_to_env,
                    max_seq_len=master_config["policy"]["max_total_sequence_length"],
                    max_rollout_turns=master_config["distillation"]["max_rollout_turns"],
                    greedy=False,
                )
            else:
                val_batch, gen_metrics = run_multi_turn_rollout(
                    policy_generation,
                    val_batch,
                    tokenizer,
                    val_task_to_env,
                    max_seq_len=master_config["policy"]["max_total_sequence_length"],
                    max_rollout_turns=master_config["distillation"]["max_rollout_turns"],
                    greedy=False,
                )
            rewards = val_batch["total_reward"]
            total_rewards.extend(rewards.tolist())
            total_lengths.append(gen_metrics["mean_gen_tokens_per_sample"])
            all_message_logs.extend(val_batch["message_log"])

        accuracy = sum(total_rewards) / len(total_rewards) if total_rewards else 0.0
        avg_length = sum(total_lengths) / len(total_lengths) if total_lengths else 0.0

        try:
            print_message_log_samples(
                all_message_logs,
                total_rewards,
                num_samples=min(
                    master_config["logger"]["num_val_samples_to_print"],
                    len(all_message_logs),
                ),
                step=step,
            )
        except Exception as e:
            print(f"  ⚠️ Error displaying message samples: {e}", flush=True)

    val_metrics = {"accuracy": accuracy, "avg_length": avg_length}
    timing_metrics = timer.get_timing_metrics(reduction_op="sum")
    timer.reset()
    return val_metrics, timing_metrics
