"""Strongly typed application configuration schemas using Pydantic.

Provides schema validation, field boundaries, and type safety for config.json.
Maintains full backward compatibility with legacy dictionary access via to_dict().
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class CaptureConfig(BaseModel):
    """Minimap capture and monitor settings."""

    model_config = ConfigDict(extra="allow")

    monitor: int = Field(default=0, ge=0, description="Monitor index for screen capture")
    fps: int = Field(default=10, ge=1, le=60, description="Capture cadence in frames per second")
    mmap_roi: list[int] = Field(
        default_factory=lambda: [45, 1009, 336, 277],
        description="Minimap ROI box [x, y, width, height] on monitor",
    )

    @field_validator("mmap_roi")
    @classmethod
    def validate_roi(cls, v: list[int]) -> list[int]:
        if len(v) != 4:
            raise ValueError("mmap_roi must have exactly 4 elements [x, y, w, h]")
        _, _, w, h = v
        if w < 10 or h < 10:
            raise ValueError(f"mmap_roi dimensions too small: w={w}, h={h} (minimum 10x10)")
        return v


class LocatorConfig(BaseModel):
    """Vision matcher and localization parameters."""

    model_config = ConfigDict(extra="allow")

    local_radius: int = Field(default=450, ge=50, description="Tracking search radius (px)")
    radius_growth: float = Field(default=1.6, ge=1.0, le=5.0)
    global_max_features: int = Field(default=100000, ge=1000)
    min_pose_scale: float = Field(default=0.25, gt=0, le=0.7)
    max_kp_frame: int = Field(
        default=1200,
        ge=100,
        le=6000,
        description="Max SIFT keypoints kept from a captured frame (response-ranked)",
    )
    fast_clahe: bool = Field(
        default=False,
        description="Try a CLAHE-preprocessed fast pass first, fall back to the "
        "default normalization when it finds no pose",
    )
    ratio: float = Field(default=0.8, ge=0.1, le=1.0)
    min_inl: int = Field(default=4, ge=1)
    min_inl_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    track_radius: float = Field(default=900.0, ge=50.0)
    ratio_local: float = Field(default=0.85, ge=0.1, le=1.0)
    min_inl_local: int = Field(default=4, ge=1)
    min_inl_rate_local: float = Field(default=0.0, ge=0.0, le=1.0)
    ratio_global: float = Field(default=0.9, ge=0.1, le=1.0)
    min_inl_global: int = Field(default=5, ge=1)
    min_inl_rate_global: float = Field(default=0.0, ge=0.0, le=1.0)
    vote_need: int = Field(default=3, ge=1)
    vote_frames: int = Field(default=5, ge=1)
    vote_radius_px: int = Field(default=300, ge=10)
    jump_gate_px: int = Field(default=3000, ge=100)
    heading_gate_deg: float = Field(default=0.0, ge=0.0, le=180.0)
    vote_inl_skip: int = Field(default=40, ge=1)
    hold_frames: int = Field(default=5, ge=0)


class MapConfig(BaseModel):
    """Active map metadata and display scaling."""

    model_config = ConfigDict(extra="allow")

    name: str = Field(default="zestafona", description="Map name prefix")
    file: str = Field(default="zestafona_map.png", description="Base map file name in data/maps/")
    thumb_factor: int = Field(default=8, ge=1)
    size: int = Field(default=32768, ge=1024)
    gray_conv: str = Field(default="luma")
    gray_gamma: float = Field(default=1.0, ge=0.1, le=3.0)
    mini_scale: float = Field(
        default=2.6544, ge=0.1, le=10.0, description="Minimap to native map pixel scale"
    )


class NavigatorConfig(BaseModel):
    """Steering and route follower parameters."""

    model_config = ConfigDict(extra="allow")

    key_source: Literal["software", "arduino"] = Field(
        default="software", description="Windows software keyboard or Arduino HID keyboard"
    )
    port: str = Field(default="COM6", description="Serial port for Arduino Micro")
    arrive_r: float = Field(default=25.0, ge=1.0, description="Waypoint arrival radius (px)")
    slow_r: float = Field(default=350.0, ge=1.0, description="Deceleration radius (px)")
    dead: float = Field(default=3.0, ge=0.0, description="Steering dead-zone (deg)")
    turn_deg: float = Field(
        default=25.0,
        ge=1.0,
        description="Heading error threshold for continuous steering (deg)",
    )
    hold_max: float = Field(
        default=8.0,
        ge=0.5,
        description="Safety timeout for continuous steering (s)",
    )
    speed_cap_kmh: float = Field(default=79.0, ge=1.0, description="Maximum driving speed in km/h")
    xte_m: float = Field(default=4.0, ge=0.0, description="Cross-track error limit (meters)")
    poll: float = Field(default=0.033, ge=0.001, description="Navigation tick rate in seconds")
    brake_d: float = Field(default=260.0, ge=1.0, description="Braking distance threshold (px)")
    dead_off: float | None = Field(default=None, description="Steering release dead-zone (deg)")
    stop_speed_kmh: float = Field(
        default=2.0,
        ge=0.5,
        description="Final-waypoint full-stop speed threshold (km/h, px floor applies)",
    )
    stop_hold: float = Field(
        default=0.8,
        ge=0.1,
        description="Seconds below stop speed before the autopilot disables itself",
    )
    stop_timeout: float = Field(
        default=5.0,
        ge=1.0,
        description="Hard timeout for final-waypoint braking before autopilot off (s)",
    )
    debug: bool = Field(default=False, description="Enable verbose nav logging")
    last_preset: str = Field(default="")


class WindowConfig(BaseModel):
    """Tkinter window geometry and state."""

    model_config = ConfigDict(extra="allow")

    geometry: str = Field(default="1030x932+3374+396")
    zoomed: bool = Field(default=False)


class DebugConfig(BaseModel):
    """Debugging options."""

    model_config = ConfigDict(extra="allow")

    collect_fail_logs: bool = Field(default=False)


class AppConfig(BaseModel):
    """Root configuration for WARDOGS autopilot."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    capture: CaptureConfig = Field(default_factory=CaptureConfig)
    locator: LocatorConfig = Field(default_factory=LocatorConfig)
    map: MapConfig = Field(default_factory=MapConfig)
    navigator: NavigatorConfig = Field(default_factory=NavigatorConfig)
    window: WindowConfig = Field(default_factory=WindowConfig)
    debug: DebugConfig = Field(default_factory=DebugConfig)
    cfg_path: str = Field(default="config.json", alias="_cfg_path")

    def to_dict(self) -> dict[str, Any]:
        """Serialize configuration to a standard Python dictionary."""
        return self.model_dump(by_alias=True)

    @classmethod
    def load(cls, path: str | Path = "config.json") -> AppConfig:
        """Load and validate configuration from a JSON file."""
        cfg_path = Path(path)
        with open(cfg_path, encoding="utf-8") as f:
            data = json.load(f)
        cfg = cls.model_validate(data)
        cfg.cfg_path = str(cfg_path)
        return cfg

    def save(self, path: str | Path | None = None) -> None:
        """Save configuration back to JSON file."""
        target_path = Path(path or self.cfg_path)
        with open(target_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
