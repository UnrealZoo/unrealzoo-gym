"""Shared MicroDuck UnrealCV protocol helpers for scenario examples."""
import ctypes
import os
import select
import sys
import time

import numpy as np


MICRODUCK_BP_PATH = "/Game/robot-biped-microduck/BP_MicroDuck.BP_MicroDuck"


def request(client, command, timeout=120, verbose=True):
    response = str(client.request(command, timeout=timeout)).strip()
    if verbose:
        print("CMD|{}".format(command))
        print("RES|{}".format(response))
    return response


def parse_vector(response, context=""):
    return tuple(
        float(value) for value in response.replace(",", " ").split()
    )


def encode_targets(targets):
    values = np.asarray(targets, dtype=np.float32).reshape(-1)
    return ",".join("{:.9g}".format(float(value)) for value in values)


def validate_bridge_contract(state):
    print(
        "BRIDGE|actuator_model={}|environment_geoms={}|dynamic_support={}".format(
            state["actuator_model"],
            state["environment_geom_count"],
            state["dynamic_support_heightfield"],
        )
    )


def read_windows_command(velocity, yaw_rate):
    key_state = ctypes.windll.user32.GetAsyncKeyState
    pressed = lambda key: bool(key_state(key) & 0x8000)
    exit_requested = pressed(ord("X")) or pressed(0x1B)
    stop = exit_requested or pressed(0x20)
    command = (
        velocity * (int(pressed(ord("I"))) - int(pressed(ord("K")))),
        0.0,
        yaw_rate * (int(pressed(ord("J"))) - int(pressed(ord("L")))),
    )
    if stop:
        command = (0.0, 0.0, 0.0)
    return command, exit_requested


class PosixKeyboard:
    def __init__(self):
        self.fd = None
        self.terminal_settings = None
        self.command = (0.0, 0.0, 0.0)
        self.last_key_time = 0.0

    def __enter__(self):
        import termios
        import tty

        self.fd = sys.stdin.fileno()
        self.terminal_settings = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        return self

    def __exit__(self, *args):
        import termios

        termios.tcsetattr(
            self.fd, termios.TCSADRAIN, self.terminal_settings
        )

    def read(self, velocity, yaw_rate):
        commands = {
            "i": (velocity, 0.0, 0.0),
            "k": (-velocity, 0.0, 0.0),
            "j": (0.0, 0.0, yaw_rate),
            "l": (0.0, 0.0, -yaw_rate),
            " ": (0.0, 0.0, 0.0),
        }
        exit_requested = False
        while select.select([sys.stdin], [], [], 0.0)[0]:
            key = sys.stdin.read(1).casefold()
            if key in commands:
                self.command = commands[key]
            if key in ("x", ""):
                self.command = (0.0, 0.0, 0.0)
                exit_requested = True
            self.last_key_time = time.perf_counter()
        if time.perf_counter() - self.last_key_time > 0.12:
            self.command = (0.0, 0.0, 0.0)
        return self.command, exit_requested
