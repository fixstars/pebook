# Copyright (c) Fixstars. All rights reserved.
import functools
import sys
from mmengine.runner import Runner
from mmengine.hooks import Hook
from mmengine.hooks.hook import DATA_BATCH
from typing import Optional, Sequence, Union


from mmdet3d.registry import HOOKS
import nvtx
import torch
import torch.distributed
import torch.utils.data


def _xor_shift(seed: int) -> int:
    seed ^= seed << 13
    seed &= 0xFFFFFFFF
    seed ^= seed >> 17
    seed &= 0xFFFFFFFF
    seed ^= seed << 5
    seed &= 0xFFFFFFFF
    return seed


def _deterministic_hash(seed: str) -> int:
    seed_bytes = seed.encode("utf-8")
    return sum(map(_xor_shift, seed_bytes))


def _get_color(seed: str) -> str:
    _COLOR_TABLE = (
        "green",
        "blue",
        "yellow",
        "purple",
        "rapids",
        "cyan",
        "red",
        "white",
        "darkgreen",
        "orange",
    )
    return _COLOR_TABLE[_deterministic_hash(seed) % len(_COLOR_TABLE)]


def _range_push(module, input) -> None:
    msg = str(type(module))
    nvtx.push_range(msg, color=_get_color(msg))
    return None


def _range_pop(module, input, output) -> None:
    nvtx.pop_range()
    return None

def _get_nvtx_function(function, message = None, color = None, domain = None, category = None, payload = None):
    if isinstance(message, str) and color is None:
        color = _get_color(message)

    @functools.wraps(function)
    def _nvtx_function(*args, **kwargs):
        with nvtx.annotate(message, color, domain, category, payload):
            return function(*args, **kwargs)

    return _nvtx_function


@HOOKS.register_module()
class NVTXHook(Hook):
    """A hook that logs the training speed of each epch."""

    priority = "NORMAL"

    def __init__(
        self,
        start_iter: int = 150,
        end_iter: int = 158,
        use_torch_emit_nvtx = True,
        capture_range_end_shutdown=True,
    ):
        super().__init__()
        self.start_iter = start_iter
        self.end_iter = end_iter
        self.capture_range_end_shutdown = capture_range_end_shutdown
        self._emit_nvtx = torch.autograd.profiler.emit_nvtx() if use_torch_emit_nvtx else None

    def before_run(self, runner: Runner) -> None:
        torch.nn.modules.module.register_module_forward_pre_hook(_range_push)
        torch.nn.modules.module.register_module_forward_hook(_range_pop)

    def _before_epoch(self, runner, mode: str = "train") -> None:
        self._iter_count = 0
        nvtx.push_range(f"{mode}_epoch{runner.epoch}", color="green")

    def _before_iter(
        self, runner, batch_idx: int, data_batch: DATA_BATCH = None, mode: str = "train"
    ) -> None:
        self._iter_count += 1
        if self.start_iter == self._iter_count:
            torch.cuda.profiler.start()
            if self._emit_nvtx is not None:
                self._emit_nvtx.__enter__()
            nvtx.push_range("target_iter", color=_get_color("target_iter"))
        if self.start_iter <= self._iter_count < self.end_iter:
            nvtx.push_range(f"iter{self._iter_count}", color="yellow")
        elif self._iter_count > self.end_iter and self.capture_range_end_shutdown:
            if torch.distributed.is_available():
                torch.distributed.destroy_process_group()
            sys.exit()

    def _after_iter(
        self,
        runner,
        batch_idx: int,
        data_batch: DATA_BATCH = None,
        outputs: Optional[Union[Sequence, dict]] = None,
        mode: str = "train",
    ) -> None:
        if self.start_iter <= self._iter_count < self.end_iter:
            nvtx.pop_range()  # iter
        elif self._iter_count == self.end_iter:
            nvtx.pop_range()  # target_iter
            if self._emit_nvtx is not None:
                self._emit_nvtx.__exit__(None, None, None)
            torch.cuda.profiler.stop()

    def _after_epoch(self, runner, mode: str = "train") -> None:
        if self._iter_count < self.end_iter:
            torch.cuda.nvtx.range_pop()
            torch.cuda.profiler.stop()
        nvtx.pop_range()  # epoch
