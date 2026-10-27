# Copyright 2026 Rebellions Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from vllm.sequence import IntermediateTensors


class PrecomputedRope:
    """cos/sin rows for the current forward, gathered on the host by the runner.

    The model wrapper sets them from its graph inputs before calling the model and
    clears them after, so RoPE reads graph inputs instead of gathering from the
    cache inside the graph. Only valid when the model has one RoPE table.
    """

    cos: torch.Tensor | None = None
    sin: torch.Tensor | None = None


PRECOMPUTED_ROPE = PrecomputedRope()


def _same_width_int(dtype: torch.dtype) -> torch.dtype:
    """Integer dtype of the same width, so numpy can gather bf16 rows as raw bits."""
    return {1: torch.int8, 2: torch.int16, 4: torch.int32, 8: torch.int64}[dtype.itemsize]


@dataclass(frozen=True)
class InputLayout:
    num_reqs: int
    num_reqs_padded: int
    query_len: int
    query_len_padded: int
    input_pad_value: int = 0
    position_pad_value: int = 0
    hidden_state_pad_value: float = 0.0
    token_index_pad_value: int = 0

    @property
    def shape(self) -> tuple[int, int]:
        return (self.num_reqs_padded, self.query_len_padded)


@dataclass
class InputBuffer:
    input_ids: torch.Tensor
    positions: torch.Tensor


@dataclass(slots=True)
class StagedModelInputs:
    input_ids: torch.Tensor
    positions: torch.Tensor
    intermediate_tensors: IntermediateTensors | None
    inputs_embeds: torch.Tensor | None
    token_indices: torch.Tensor | None
    # For Eagle3 drafter
    hidden_states: torch.Tensor | None = None
    # RoPE rows gathered on the host (VLLM_RBLN_HOST_ROPE)
    rope_cos: torch.Tensor | None = None
    rope_sin: torch.Tensor | None = None

    def as_kwargs(self) -> dict[str, Any]:
        kwargs = {
            "input_ids": self.input_ids,
            "positions": self.positions,
            "intermediate_tensors": self.intermediate_tensors,
            "inputs_embeds": self.inputs_embeds,
            "token_indices": self.token_indices,
        }
        if self.rope_cos is not None:
            kwargs["rope_cos"] = self.rope_cos
            kwargs["rope_sin"] = self.rope_sin
        return kwargs


class InputStager:
    def __init__(self, device: torch.device):
        self.device = device
        self._buffers: dict[tuple, InputBuffer] = {}
        self._hidden_state_buffers: dict[tuple, torch.Tensor] = {}
        self._token_indices_buffers: dict[tuple[torch.dtype, int], torch.Tensor] = {}
        self._rope_tables: tuple[np.ndarray, np.ndarray, torch.dtype] | None = None
        self._rope_buffers: dict[tuple, tuple] = {}
        self._embed_table: tuple[np.ndarray, torch.dtype] | None = None
        self._embed_buffers: dict[tuple, tuple] = {}

    def set_embed_table(self, weight: torch.Tensor) -> None:
        """Host view of the token embedding table [vocab, hidden], for prefill
        inputs_embeds gathered by numpy on one thread (VLLM_RBLN_HOST_EMBED)."""
        dtype = weight.dtype
        self._embed_table = (
            weight.detach().to("cpu").contiguous().view(_same_width_int(dtype)).numpy(),
            dtype,
        )

    def set_rope_tables(
        self, cos: torch.Tensor, sin: torch.Tensor, dtype: torch.dtype
    ) -> None:
        """Host copies of the RoPE cos/sin caches [max_pos, rot_dim], in the graph's
        dtype, kept as same-width integer arrays so numpy can gather them on one
        thread (no torch intra-op pool wake-up)."""
        as_np = lambda t: (
            t.detach().to("cpu", dtype).contiguous().view(_same_width_int(dtype)).numpy()
        )
        self._rope_tables = (as_np(cos), as_np(sin), dtype)

    def stage(
        self,
        *,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        token_indices: torch.Tensor | None = None,
        hidden_states: torch.Tensor | None = None,
        layout: InputLayout,
        host_embed: bool = False,
    ) -> StagedModelInputs:
        buf = self._get_or_create_buffer(layout, input_ids, positions)

        assert buf.input_ids.shape == layout.shape
        buf.input_ids.fill_(layout.input_pad_value)
        buf.input_ids[: layout.num_reqs, : layout.query_len].copy_(
            input_ids,
            non_blocking=True,
        )

        buf.positions.fill_(layout.position_pad_value)
        buf.positions[: layout.num_reqs, : layout.query_len].copy_(
            positions,
            non_blocking=True,
        )

        rope_cos, rope_sin = self._stage_rope(positions, layout)
        if host_embed and self._embed_table is not None and inputs_embeds is None:
            inputs_embeds = self._stage_embeds(input_ids, layout)
        return StagedModelInputs(
            input_ids=buf.input_ids,
            positions=buf.positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            token_indices=self._stage_token_indices(token_indices, layout),
            hidden_states=self._stage_hidden_states(hidden_states, layout),
            rope_cos=rope_cos,
            rope_sin=rope_sin,
        )

    def _stage_embeds(
        self,
        input_ids: torch.Tensor,
        layout: InputLayout,
    ) -> torch.Tensor:
        table, dtype = self._embed_table
        key = (layout.shape, dtype)
        if (bufs := self._embed_buffers.get(key)) is None:
            shape = (*layout.shape, table.shape[-1])
            bufs = (
                np.zeros(layout.shape, dtype=np.int64),
                np.empty(shape, dtype=table.dtype),
                torch.empty(shape, dtype=dtype, device=self.device),
            )
            self._embed_buffers[key] = bufs
        ids, host, dev = bufs
        ids.fill(layout.input_pad_value)
        ids[: layout.num_reqs, : layout.query_len] = input_ids.numpy()
        np.take(table, ids, axis=0, out=host)
        dev.copy_(torch.from_numpy(host).view(dtype), non_blocking=True)
        return dev

    def _stage_rope(
        self,
        positions: torch.Tensor,
        layout: InputLayout,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if self._rope_tables is None:
            return None, None
        cos_tab, sin_tab, dtype = self._rope_tables
        key = (layout.shape, dtype)
        if (bufs := self._rope_buffers.get(key)) is None:
            shape = (*layout.shape, cos_tab.shape[-1])
            # host staging rows (persistent, so a non-blocking copy never reads
            # freed memory) and the device inputs
            host = [np.empty(shape, dtype=cos_tab.dtype) for _ in range(2)]
            dev = [torch.empty(shape, dtype=dtype, device=self.device) for _ in range(2)]
            bufs = (np.zeros(layout.shape, dtype=np.int64), host, dev)
            self._rope_buffers[key] = bufs
        pos, host, dev = bufs
        pos.fill(layout.position_pad_value)
        pos[: layout.num_reqs, : layout.query_len] = positions.numpy()
        for tab, h, d in zip((cos_tab, sin_tab), host, dev):
            np.take(tab, pos, axis=0, out=h)
            d.copy_(torch.from_numpy(h).view(dtype), non_blocking=True)
        return dev[0], dev[1]

    def _get_or_create_buffer(
        self,
        layout: InputLayout,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> InputBuffer:
        key = (
            layout.shape,
            input_ids.dtype,
            positions.dtype,
        )
        if key in self._buffers:
            return self._buffers[key]

        buf = InputBuffer(
            input_ids=torch.empty(
                layout.shape,
                dtype=input_ids.dtype,
                device=self.device,
            ),
            positions=torch.empty(
                layout.shape,
                dtype=positions.dtype,
                device=self.device,
            ),
        )
        self._buffers[key] = buf
        return buf

    def _stage_hidden_states(
        self,
        hidden_states: torch.Tensor | None,
        layout: InputLayout,
    ) -> torch.Tensor | None:
        if hidden_states is None:
            return None

        key = (layout.shape, hidden_states.dtype, hidden_states.shape[-1])
        if (buf := self._hidden_state_buffers.get(key)) is None:
            buf = torch.empty(
                (*layout.shape, hidden_states.shape[-1]),
                dtype=hidden_states.dtype,
                device=self.device,
            )
            self._hidden_state_buffers[key] = buf

        buf[layout.num_reqs :].fill_(layout.hidden_state_pad_value)
        buf[: layout.num_reqs, layout.query_len :].fill_(layout.hidden_state_pad_value)
        buf[: layout.num_reqs, : layout.query_len].copy_(
            hidden_states,
            non_blocking=True,
        )
        return buf

    def _stage_token_indices(
        self,
        token_indices: torch.Tensor | None,
        layout: InputLayout,
    ) -> torch.Tensor | None:
        if token_indices is None:
            return None

        num_indices = token_indices.shape[0]
        assert num_indices <= layout.num_reqs_padded

        key = (token_indices.dtype, layout.num_reqs_padded)
        if (buf := self._token_indices_buffers.get(key)) is None:
            buf = torch.empty(
                layout.num_reqs_padded,
                dtype=token_indices.dtype,
                device=self.device,
            )
            self._token_indices_buffers[key] = buf

        buf.fill_(layout.token_index_pad_value)
        buf[:num_indices].copy_(token_indices, non_blocking=True)
        return buf
