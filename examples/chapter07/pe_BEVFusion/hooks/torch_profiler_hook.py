# Copyright (c) Fixstars. All rights reserved.
import sys
import pickle
from mmengine.hooks import Hook
from mmengine.hooks.hook import DATA_BATCH
from typing import Optional, Sequence, Union


from mmdet3d.registry import HOOKS
import torch


@HOOKS.register_module()
class TorchProfilerHook(Hook):
    """A hook that logs the training speed of each epch."""

    priority = "NORMAL"

    def __init__(self):
        super().__init__()
        self._tensorboard_trace_handler = torch.profiler.tensorboard_trace_handler("./profile")
        self._profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            with_stack=True,
            with_flops=True,
            with_modules=True,
            schedule=torch.profiler.schedule(wait=45, warmup=5, active=2, repeat=1),
            on_trace_ready=self.trace_handler,
        )


    def trace_handler(self, p):
        self._tensorboard_trace_handler(p)
        p.export_chrome_trace("./profile/torch_" + str(p.step_num) + ".json")

    def _before_epoch(self, runner, mode: str = "train") -> None:
        self._profiler.start()

    def _after_iter(
        self,
        runner,
        batch_idx: int,
        data_batch: DATA_BATCH = None,
        outputs: Optional[Union[Sequence, dict]] = None,
        mode: str = "train",
    ) -> None:
        self._profiler.step()

    def _after_epoch(self, runner, mode: str = "train") -> None:
        self._profiler.stop()
        if torch.distributed.is_available():
            torch.distributed.destroy_process_group()
        sys.exit()
