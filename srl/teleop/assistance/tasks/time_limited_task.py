# Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the MIT License [see LICENSE for details].


import time
from typing import Optional


class TimeLimitedTask():
    """[summary]

        Args:
            name (str, optional): [description].
            target_position (Optional[np.ndarray], optional): [description]. Defaults to None.
            cube_size (Optional[np.ndarray], optional): [description]. Defaults to None.
            offset (Optional[np.ndarray], optional): [description]. Defaults to None.
        """

    def __init__(
        self,
        max_duration: Optional[int]
    ) -> None:
        self.max_duration = max_duration
        self._start_wallclock_stamp = None
        self._done = False

    def get_params(self) -> dict:
        non_optional = self.max_duration if self.max_duration is not None else -1
        return {
            "max_duration": non_optional,
        }

    @property
    def time_remaining(self):
        if not self.max_duration:
            return None
        return self.max_duration - (time.time() - self._start_wallclock_stamp)

    def is_done(self):
        return self._done

    def pre_step(self, time_step_index: int, simulation_time: float) -> None:
        now = time.time()
        if self._start_wallclock_stamp is None:
            self._start_wallclock_stamp = time.time()
        if self.max_duration and now - self._start_wallclock_stamp > self.max_duration:
            self._done = True
