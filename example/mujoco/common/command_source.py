"""Shared I/J/K/L and fixed velocity command sources."""
import ctypes
import os
import select
import sys
import time

import numpy as np


def ijkl_command(i_pressed, k_pressed, j_pressed, l_pressed, velocity, yaw_rate):
    return (
        velocity * (int(bool(i_pressed)) - int(bool(k_pressed))),
        0.0,
        yaw_rate * (int(bool(j_pressed)) - int(bool(l_pressed))),
    )


def interpolate_command(start, target, elapsed, ramp_seconds):
    if ramp_seconds <= 0.0:
        return target
    alpha = min(1.0, max(0.0, elapsed / ramp_seconds))
    return tuple(
        start[index] + (target[index] - start[index]) * alpha
        for index in range(3)
    )


class HoldKeyboardController:
    def __init__(self):
        self.fd = None
        self.terminal_settings = None

    def __enter__(self):
        if os.name != "nt":
            import termios
            import tty

            self.fd = sys.stdin.fileno()
            self.terminal_settings = termios.tcgetattr(self.fd)
            tty.setcbreak(self.fd)
        return self

    def __exit__(self, *args):
        if os.name != "nt":
            import termios

            termios.tcsetattr(
                self.fd, termios.TCSADRAIN, self.terminal_settings
            )

    def read(self, current_command, velocity, yaw_rate):
        if os.name == "nt":
            key_state = ctypes.windll.user32.GetAsyncKeyState
            pressed = lambda key: bool(key_state(key) & 0x8000)
            exit_requested = pressed(ord("X")) or pressed(0x1B)
            hard_stop = exit_requested or pressed(0x20)
            command = ijkl_command(
                pressed(ord("I")),
                pressed(ord("K")),
                pressed(ord("J")),
                pressed(ord("L")),
                velocity,
                yaw_rate,
            )
            if hard_stop:
                command = (0.0, 0.0, 0.0)
            return command, exit_requested, hard_stop

        command = current_command
        exit_requested = False
        hard_stop = False
        commands = {
            "i": (velocity, 0.0, 0.0),
            "k": (-velocity, 0.0, 0.0),
            "j": (0.0, 0.0, yaw_rate),
            "l": (0.0, 0.0, -yaw_rate),
            " ": (0.0, 0.0, 0.0),
        }
        while select.select([sys.stdin], [], [], 0.0)[0]:
            key = sys.stdin.read(1).casefold()
            if key in commands:
                command = commands[key]
                hard_stop = key == " "
            if key in ("x", ""):
                command = (0.0, 0.0, 0.0)
                exit_requested = True
                hard_stop = True
        return command, exit_requested, hard_stop


class FixedCommand:
    def __init__(self, command):
        self.command = np.asarray(command, dtype=np.float32)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def read(self):
        return self.command, False


class KeyboardCommand:
    def __init__(self, velocity, yaw_rate):
        self.velocity = velocity
        self.yaw_rate = yaw_rate
        self.command = np.zeros(3, dtype=np.float32)
        self.last_key_time = 0.0
        self.fd = None
        self.terminal_settings = None

    def __enter__(self):
        if os.name != "nt":
            import termios
            import tty

            self.fd = sys.stdin.fileno()
            self.terminal_settings = termios.tcgetattr(self.fd)
            tty.setcbreak(self.fd)
        return self

    def __exit__(self, *args):
        if os.name != "nt":
            import termios

            termios.tcsetattr(
                self.fd, termios.TCSADRAIN, self.terminal_settings
            )

    def _read_windows(self):
        key_state = ctypes.windll.user32.GetAsyncKeyState
        pressed = lambda key: bool(key_state(key) & 0x8000)
        command = np.asarray(
            [
                self.velocity
                * (int(pressed(ord("I"))) - int(pressed(ord("K")))),
                0.0,
                self.yaw_rate
                * (int(pressed(ord("J"))) - int(pressed(ord("L")))),
            ],
            dtype=np.float32,
        )
        stop = pressed(0x20)
        exit_requested = pressed(ord("X")) or pressed(0x1B)
        if stop or exit_requested:
            command.fill(0.0)
        return command, exit_requested

    def _read_posix(self):
        commands = {
            "i": np.asarray([self.velocity, 0.0, 0.0], dtype=np.float32),
            "k": np.asarray([-self.velocity, 0.0, 0.0], dtype=np.float32),
            "j": np.asarray([0.0, 0.0, self.yaw_rate], dtype=np.float32),
            "l": np.asarray([0.0, 0.0, -self.yaw_rate], dtype=np.float32),
            " ": np.zeros(3, dtype=np.float32),
        }
        exit_requested = False
        while select.select([sys.stdin], [], [], 0.0)[0]:
            key = sys.stdin.read(1).casefold()
            if key in commands:
                self.command = commands[key]
            if key in ("x", ""):
                self.command.fill(0.0)
                exit_requested = True
            self.last_key_time = time.perf_counter()
        if time.perf_counter() - self.last_key_time > 0.12:
            self.command.fill(0.0)
        return self.command, exit_requested

    def read(self):
        if os.name == "nt":
            return self._read_windows()
        return self._read_posix()
