# Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the MIT License [see LICENSE for details].

from time import perf_counter
import carb.profiler


profile_table = {}
name_stack = []
class profile:
    def __init__(self, name="", active=True) -> None:
        self.name = name
        self.active = active
        pass
    def __enter__(self):
        self.time = perf_counter()
        carb.profiler.begin(1, self.name, active=self.active)
        return self

    def __exit__(self, type, value, traceback):
        self.time = perf_counter() - self.time
        self.readout = f'{self.name} Time: {self.time * 1000:.2f} milliseconds'
        carb.profiler.end(1, self.active)


def is_profiler_active():
    # Flip this to True if you want profiling information to print out
    return False


def begin(mask, name, stack_offset=0, active=False):
    if not is_profiler_active() or not active:
        return
    profile_table[name] = perf_counter()
    name_stack.append(name)


def end(mask, active=False):
    if not is_profiler_active() or not active:
        return
    start_stack_depth = len(name_stack)
    if start_stack_depth == 0:
        return
    name = name_stack.pop()
    print("  " * (start_stack_depth - 1) + f"{name}: {(perf_counter() - profile_table[name]) * 1000:.2f}ms")
    del profile_table[name]

# TBR
carb.profiler.begin = begin
carb.profiler.end = end
