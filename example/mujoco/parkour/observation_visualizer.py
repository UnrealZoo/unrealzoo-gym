#!/usr/bin/env python3
"""Live diagnostics for the Go1 Robot Parkour policy observation."""
from __future__ import annotations

import json
import math
import time
from pathlib import Path


class Go1ObservationVisualizer:
    """Render and persist the exact depth/proprioception used by the policy."""

    WINDOW_NAME = "Go1 Robot Parkour Observation"

    def __init__(
        self,
        np_module,
        enabled: bool,
        dump_dir: str | Path,
        dump_every: int,
        depth_max: float,
        expected_raw_shape: tuple[int, int],
        expected_processed_shape: tuple[int, int],
    ):
        self.np = np_module
        self.enabled = bool(enabled)
        self.dump_dir = Path(dump_dir).expanduser().resolve()
        self.dump_every = max(0, int(dump_every))
        self.depth_max = float(depth_max)
        self.expected_raw_shape = tuple(expected_raw_shape)
        self.expected_processed_shape = tuple(expected_processed_shape)
        self.cv2 = None
        self.previous_depth = None
        self.previous_depth_frame = None
        self.frozen_frames = 0
        self.last_warning = ""
        self.last_warning_time = -math.inf
        self.last_auto_dump_time = -math.inf
        self.last_dashboard = None
        self.last_raw_depth = None
        self.last_processed_depth = None
        self.last_metadata = None
        self._last_update_depth_frame = None

        if self.enabled:
            try:
                import cv2

                self.cv2 = cv2
                cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(self.WINDOW_NAME, 1280, 720)
            except Exception as exc:
                self.enabled = False
                self.cv2 = None
                print(f"OBS_WINDOW|enabled=no|reason={type(exc).__name__}:{exc}")

        if self.enabled or self.dump_every > 0:
            self.dump_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"OBS_VISUALIZER|window={'yes' if self.enabled else 'no'}|"
            f"dump_dir={self.dump_dir}|dump_every={self.dump_every}"
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def _processed_numpy(self, processed_depth):
        if hasattr(processed_depth, "detach"):
            processed = processed_depth.detach().cpu().numpy()
        else:
            processed = self.np.asarray(processed_depth)
        processed = self.np.asarray(processed, dtype=self.np.float32).squeeze()
        if processed.ndim != 2:
            raise ValueError(f"Expected processed depth HxW, got {processed.shape}")
        return processed

    def _depth_health(self, raw_depth, depth_frame, moving):
        np = self.np
        raw = np.asarray(raw_depth, dtype=np.float32)
        finite = np.isfinite(raw)
        finite_values = raw[finite]
        nonfinite_ratio = 1.0 - float(finite.mean()) if raw.size else 1.0
        if finite_values.size:
            percentiles = np.percentile(finite_values, [0, 5, 50, 95, 100]).tolist()
            zero_ratio = float((finite_values <= 0.01).mean())
            very_near_10cm_ratio = float((finite_values < 0.10).mean())
            very_near_20cm_ratio = float((finite_values < 0.20).mean())
            near_ratio = float((finite_values < self.depth_max).mean())
            far_ratio = float((finite_values >= self.depth_max - 0.01).mean())
        else:
            percentiles = [math.nan] * 5
            zero_ratio = 1.0
            very_near_10cm_ratio = 0.0
            very_near_20cm_ratio = 0.0
            near_ratio = 0.0
            far_ratio = 0.0

        frame_delta = math.nan
        if depth_frame != self.previous_depth_frame:
            comparable = self.previous_depth is not None and self.previous_depth.shape == raw.shape
            if comparable:
                current_safe = np.nan_to_num(raw, nan=self.depth_max, posinf=self.depth_max)
                previous_safe = np.nan_to_num(
                    self.previous_depth, nan=self.depth_max, posinf=self.depth_max
                )
                frame_delta = float(np.mean(np.abs(current_safe - previous_safe)))
                if moving and frame_delta < 1e-4:
                    self.frozen_frames += 1
                else:
                    self.frozen_frames = 0
            self.previous_depth = raw.copy()
            self.previous_depth_frame = depth_frame

        warnings = []
        if tuple(raw.shape) != self.expected_raw_shape:
            warnings.append(f"raw_shape={tuple(raw.shape)}")
        if nonfinite_ratio > 0.0:
            warnings.append(f"nonfinite={nonfinite_ratio:.1%}")
        if zero_ratio > 0.05:
            warnings.append(f"zero_depth={zero_ratio:.1%}")
        if very_near_10cm_ratio > 0.01:
            warnings.append(
                f"possible_self_occlusion_lt10cm={very_near_10cm_ratio:.1%}"
            )
        if near_ratio < 0.02:
            warnings.append(f"no_geometry_under_{self.depth_max:.1f}m")
        if self.frozen_frames >= 5:
            warnings.append(f"frozen_frames={self.frozen_frames}")

        return {
            "percentiles_m": percentiles,
            "nonfinite_ratio": nonfinite_ratio,
            "zero_ratio": zero_ratio,
            "very_near_10cm_ratio": very_near_10cm_ratio,
            "very_near_20cm_ratio": very_near_20cm_ratio,
            "near_ratio": near_ratio,
            "far_ratio": far_ratio,
            "frame_delta_m": frame_delta,
            "frozen_frames": self.frozen_frames,
            "warnings": warnings,
        }

    def _colorize_depth(self, depth, width, height, normalized=False):
        cv2 = self.cv2
        np = self.np
        values = np.asarray(depth, dtype=np.float32)
        if normalized:
            values = values * self.depth_max
        values = np.nan_to_num(values, nan=self.depth_max, posinf=self.depth_max, neginf=0.0)
        values = np.clip(values, 0.0, self.depth_max)
        intensity = ((1.0 - values / max(self.depth_max, 1e-6)) * 255.0).astype(np.uint8)
        colored = cv2.applyColorMap(intensity, cv2.COLORMAP_TURBO)
        return cv2.resize(colored, (width, height), interpolation=cv2.INTER_NEAREST)

    @staticmethod
    def _put_lines(cv2, canvas, lines, x, y, color=(230, 230, 230), scale=0.52):
        for line in lines:
            cv2.putText(
                canvas, str(line), (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, 1, cv2.LINE_AA,
            )
            y += 23

    def _draw_bars(self, canvas, values, x, y, width, height, label, limit):
        cv2 = self.cv2
        values = self.np.asarray(values, dtype=self.np.float32).reshape(-1)
        cv2.putText(
            canvas, label, (x, y - 7), cv2.FONT_HERSHEY_SIMPLEX,
            0.46, (210, 210, 210), 1, cv2.LINE_AA,
        )
        if not values.size:
            return
        center = y + height // 2
        cv2.line(canvas, (x, center), (x + width, center), (80, 80, 80), 1)
        bar_width = max(2, width // values.size)
        safe_limit = max(abs(float(limit)), 1e-6)
        for index, value in enumerate(values):
            value = max(-safe_limit, min(safe_limit, float(value)))
            end_y = int(round(center - value / safe_limit * (height // 2 - 2)))
            color = (80, 210, 120) if value >= 0.0 else (80, 140, 240)
            bx = x + index * bar_width + 1
            cv2.rectangle(
                canvas, (bx, min(center, end_y)),
                (bx + bar_width - 2, max(center, end_y)), color, -1,
            )

    def _render(self, raw, processed, metadata, health):
        cv2 = self.cv2
        canvas = self.np.zeros((720, 1280, 3), dtype=self.np.uint8)
        canvas[35:385, 10:630] = self._colorize_depth(raw, 620, 350)
        canvas[35:385, 650:1270] = self._colorize_depth(
            processed, 620, 350, normalized=True
        )
        cv2.putText(canvas, "UnrealCV raw depth (metres)", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, "Policy depth 48x64 (0..1)", (650, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)

        p = health["percentiles_m"]
        warning_text = ", ".join(health["warnings"]) if health["warnings"] else "OK"
        warning_color = (50, 70, 255) if health["warnings"] else (80, 220, 110)
        self._put_lines(cv2, canvas, [
            f"step={metadata['step']} sim={metadata['sim_time']:.3f}s depth_frame={metadata['depth_frame']}",
            f"camera={metadata['camera_id']} source={metadata['camera_source']}",
            f"raw={tuple(raw.shape)} processed={tuple(processed.shape)} age={metadata['depth_age']:.3f}s",
            f"depth min/p05/p50/p95/max={p[0]:.3f}/{p[1]:.3f}/{p[2]:.3f}/{p[3]:.3f}/{p[4]:.3f}m",
            f"near(<{self.depth_max:.1f}m)={health['near_ratio']:.1%} <10cm={health['very_near_10cm_ratio']:.1%} <20cm={health['very_near_20cm_ratio']:.1%}",
            f"zero={health['zero_ratio']:.1%} nonfinite={health['nonfinite_ratio']:.1%}",
            f"frame_delta={health['frame_delta_m']:.6f}m frozen={health['frozen_frames']}",
        ], 14, 414)
        cv2.putText(canvas, f"health: {warning_text}", (14, 581), cv2.FONT_HERSHEY_SIMPLEX, 0.58, warning_color, 2, cv2.LINE_AA)

        command = metadata["command"]
        gravity = metadata["gravity"]
        self._put_lines(cv2, canvas, [
            f"command vx/vy/yaw={command[0]:+.3f} {command[1]:+.3f} {command[2]:+.3f}",
            f"gravity={gravity[0]:+.3f} {gravity[1]:+.3f} {gravity[2]:+.3f}",
            f"contacts={metadata['contacts']}",
        ], 650, 414)
        proprio = metadata["proprio"]
        self._draw_bars(canvas, proprio[12:24], 650, 520, 300, 80, "joint position obs", 2.0)
        self._draw_bars(canvas, proprio[24:36], 970, 520, 300, 80, "joint velocity obs", 2.0)
        self._draw_bars(canvas, metadata["action"], 650, 630, 620, 75, "policy action", 2.0)
        return canvas

    def update(
        self, *, step, sim_time, depth_frame, raw_depth, processed_depth,
        proprio, command, action, payload, depth_age, camera_id, camera_source,
    ):
        raw = self.np.asarray(raw_depth, dtype=self.np.float32)
        processed = self._processed_numpy(processed_depth)
        moving = any(abs(float(value)) > 1e-5 for value in command)
        health = self._depth_health(raw, depth_frame, moving)
        if tuple(processed.shape) != self.expected_processed_shape:
            health["warnings"].append(f"processed_shape={tuple(processed.shape)}")

        gravity = payload.get("gravity", proprio[6:9])
        metadata = {
            "step": int(step),
            "sim_time": float(sim_time or 0.0),
            "depth_frame": int(depth_frame),
            "depth_age": float(depth_age),
            "camera_id": str(camera_id),
            "camera_source": str(camera_source),
            "command": [float(value) for value in command],
            "gravity": [float(value) for value in gravity],
            "contacts": payload.get("foot_contacts", []),
            "proprio": [float(value) for value in proprio],
            "action": [float(value) for value in action],
            "health": health,
        }

        is_new_frame = depth_frame != self._last_update_depth_frame
        self._last_update_depth_frame = depth_frame
        now = time.monotonic()
        if is_new_frame and health["warnings"]:
            warning = ",".join(health["warnings"])
            if warning != self.last_warning or now - self.last_warning_time >= 2.0:
                print(
                    f"OBS_WARNING|step={step}|depth_frame={depth_frame}|"
                    f"warning={warning}|p50_m={health['percentiles_m'][2]:.4f}|"
                    f"near_ratio={health['near_ratio']:.4f}|"
                    f"frame_delta_m={health['frame_delta_m']:.6f}"
                )
                self.last_warning = warning
                self.last_warning_time = now

        dashboard = self._render(raw, processed, metadata, health) if self.cv2 is not None else None
        self.last_raw_depth = raw.copy()
        self.last_processed_depth = processed.copy()
        self.last_metadata = metadata
        self.last_dashboard = dashboard

        if self.enabled and dashboard is not None:
            self.cv2.imshow(self.WINDOW_NAME, dashboard)
            self.cv2.waitKey(1)

        periodic = self.dump_every > 0 and is_new_frame and depth_frame % self.dump_every == 0
        anomaly = is_new_frame and bool(health["warnings"]) and now - self.last_auto_dump_time >= 2.0
        if periodic or anomaly:
            self.save_last("periodic" if periodic else "anomaly")
            if anomaly:
                self.last_auto_dump_time = now

    def save_last(self, reason):
        if self.last_metadata is None or self.last_raw_depth is None:
            return None
        self.dump_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        stem = f"{stamp}_step{self.last_metadata['step']:06d}_{reason}"
        metadata_path = self.dump_dir / f"{stem}.json"
        self.np.save(self.dump_dir / f"{stem}_raw.npy", self.last_raw_depth)
        self.np.save(self.dump_dir / f"{stem}_processed.npy", self.last_processed_depth)
        metadata_path.write_text(
            json.dumps(self.last_metadata, indent=2, ensure_ascii=False, allow_nan=True),
            encoding="utf-8",
        )
        if self.cv2 is not None:
            self.cv2.imwrite(str(self.dump_dir / f"{stem}_raw.png"), self._colorize_depth(self.last_raw_depth, 848, 480))
            self.cv2.imwrite(str(self.dump_dir / f"{stem}_processed.png"), self._colorize_depth(self.last_processed_depth, 640, 480, normalized=True))
            if self.last_dashboard is not None:
                self.cv2.imwrite(str(self.dump_dir / f"{stem}_dashboard.png"), self.last_dashboard)
        print(f"OBS_DUMP|reason={reason}|metadata={metadata_path}")
        return metadata_path

    def close(self):
        if self.cv2 is not None:
            try:
                self.cv2.destroyWindow(self.WINDOW_NAME)
                self.cv2.waitKey(1)
            except Exception:
                pass
