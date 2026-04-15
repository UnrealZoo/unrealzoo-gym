# Changelog

All notable changes to this project are documented in this file.

## [v3.0-stage] - 2026-04-13

This stage update focuses on task unification, mixed-population support, keyboard interaction upgrades, and demo re-organization.

### Added

- Added mixed-category population initialization in `UnrealCv_base.set_population`, supporting per-slot category lists such as `['player', 'player', 'player', 'drone']`.
- Added template-based dynamic spawn pipeline for agents:
  - environment JSON can be template-only (no pre-placed agent names),
  - runtime spawn uses `asset_path` + `spawn_from_path`,
  - agent slots are materialized during `set_population`.
- Added heterogeneous cooperation demo: `example/multi_agent/HeterogeneousCooperation/Aerial-Ground-Cooperative.py`.
  - 3 ground agents + 1 drone.
  - Drone supports `DronePoseTracker` auto-follow and keyboard override.
  - Stable per-episode formation setup, +2m drone spawn height, and 2x2 observation mosaic.
- Added keyboard animal tracking demo: `example/tracking/basic/tracking_keyboard_animal.py`.
- Added platform binary template workflow in `generate_env_config.py`:
  - `ENV_BIN_TEMPLATE_LINUX`
  - `ENV_BIN_TEMPLATE_WIN`
  - `ENV_BIN_TEMPLATE_MAC`
  - New helper flow to generate per-platform `env_bin` fields in JSON.
- Added default navigation target schema in generated JSON:
  - `env.targets.Point: []`

### Changed

- Refactored `generate_env_config.py` to support:
  - template-based binary path generation,
  - unified agent templates with `asset_path` (instead of fixed pre-placed actor entries),
  - UE5 backup field reuse (`height`, `safe_start`, `reset_area`).
- Updated base environment spawn behavior (`UnrealCv_base`):
  - supports template-only startup and delayed population spawn,
  - uses category-aware templates when spawning mixed agents,
  - updates action/observation/camera slots together with dynamic population changes.
- Updated navigation env behavior (`navigation.py`, `navigationmulti.py`):
  - no implicit target filling;
  - warning when `targets.Point` is empty;
  - safe reset/step behavior when target list is empty.
- Updated keyboard navigation mappings for human demos:
  - `F` -> open door
  - `H` -> enter/exit vehicle
  - `Ctrl` -> crouch
  - `E` -> pickup
- Updated drone keyboard demo for 4-DOF continuous control (`vx`, `vy`, `vz`, `vyaw`) and task intro display flow.
- Updated interaction animation mappings in `Character_API`:
  - `open_door` uses default `state=1`,
  - `enter_vehicle` mapped to `enter_exit_car(player_index=0)`,
  - standardized pickup action mapping.
- Reorganized `example/` into clearer task-oriented structure:
  - tracking
  - navigation
  - multi_agent
  - legacy/wip areas
- Updated `example/README.md` with new categorized demo index.
- Updated root `README.md` example commands to the new paths.

### Fixed

- Fixed interaction action mismatch causing `KeyError` for pickup animation by unifying runtime mapping.
- Fixed mixed-action schema mismatch for non-human templates by aligning config generation with required keys.
- Fixed drone cooperation demo control priority so keyboard input overrides auto tracker only while keys are pressed.

### Compatibility Notes

- Navigation tasks now expect `env.targets.Point` to exist in map JSON (can be empty list).
  - Empty list triggers a warning but does not crash; navigation reward will be 0 until targets are set.
- For generated JSONs, users should select or set navigation targets before running target-dependent navigation workflows.
- Mixed-population scenarios should explicitly set `env.unwrapped.agents_category` **before** population wrappers apply.
  - Incorrect timing: `env.reset()` → `agents_category = [...]` → wrapper (will not work)
  - Correct timing: `agents_category = [...]` → `env.reset()` → wrapper
- Older workflows that relied on fixed `agents.{name}` pre-placement should migrate to template-based `agents.{category}` + runtime population.
  - **Why migrate**: Template-based spawn decouples map JSON from instance count, enabling dynamic population changes and mixed-category experiments without editing JSON.

### Note

- Some legacy demos were moved/replaced; prefer new paths listed in `example/README.md`.

### UnrealCV+ Plugin

Summary of notable UnrealCV+ changes in v3.0 stage.

- Added runtime `PAK` mounting support for packaged applications, making it easier to extend content without rebuilding the whole project.
  - Supports mount, unmount, mounted pak listing, mount-state checks, file enumeration, asset enumeration, asset rescanning, and dynamic asset loading/registration.
  - Python APIs:
    - `mount_pak()`
    - `unmount_pak()`
    - `get_mounted_paks()`
    - `is_pak_mounted()`
    - `get_pak_files()`
    - `get_pak_assets_in_pak()`
    - `scan_pak_assets()`
    - `load_pak_asset()`
    - `get_pak_assets()`
    - `register_pak_assets()`
  - See unrealzoo website: `Home`->`Document`->`Import Custom Assets` for more information.

- Added panoramic camera support for 360-degree equirectangular image generation.
  - Supports per-camera panorama cube resolution and direct panorama export to file.
  - Python APIs:
    - `set_camera_panoramic_resolution()`
    - `capture_panoramic()`

- Added a faster C++ video recording pipeline, improving recording efficiency and making large-scale capture workflows more practical.
  - Supports direct recording from a camera or actor, configurable output directory, frame rate, duration, and selected recording channels.
  - Python APIs:
    - `start_simple_recording()`: start a recording job with output path, FPS, duration, and selected channels such as `lit` or `mask`.
    - `stop_recording()`: stop an active recording job for the target camera or capture actor.
    - `get_use_movie_quality_rendering()` / `set_use_movie_quality_rendering()`: query or change the global movie-quality rendering switch used by the recording pipeline.
    - `get_record_via_viewport()` / `set_record_via_viewport()`: query or change the global switch for whether recording uses the viewport capture path.
    - `get_warmup_frames()` / `set_warmup_frames()`: query or configure the global number of warmup frames rendered before recording starts.
    - `get_paused_tick_interval()` / `set_paused_tick_interval()`: query or configure the global paused tick interval in seconds.
    - `get_record_add_timestamp()` / `set_record_add_timestamp()`: query or control whether an active capture actor appends a timestamp suffix to recorded outputs.
    - `get_recording_paused()` / `set_recording_paused()`: query or control whether an active recording session is paused.

- Improved capture performance for common modalities such as `lit`, `mask`, and `depth`.
  - Since December 2025, frame throughput has been steadily improved, especially around the `BaseCamSensor` path.
  - In test cases, `lit` capture throughput improved from roughly `15 fps` to `20 fps`.

- Updated camera ID support to remain compatible with existing legacy camera addressing while also providing the newer stable `CID` format for long-term use in scripts and configurations.
  - Existing Python camera APIs remain backward compatible.
  - New Python APIs expose stable `CID-*` camera identifiers explicitly:
    - `get_camera_list_cid()`
    - `get_camera_id_map()`
  - Legacy camera discovery remains available through:
    - `get_camera_list_legacy()`

- Added annotation command support for scene and actor labeling workflows.
  - Supports annotating a single actor, annotating the whole world, clearing world annotation, enabling/disabling annotation cache, and clearing cached annotation components.
  - Python APIs:
    - `annotate_object()`
    - `annotate_world()`
    - `clear_world_annotation()`
    - `set_annotation_cache_enabled()`
    - `clear_annotation_cache()`

- Added object spawning from asset paths via `spawn_from_path`.
  - This complements the older class-based spawn flow and is better suited for runtime content referenced by full asset path.
  - Python APIs:
    - `spawn_object_from_path()`
  - `set_new_obj()` now gives a clearer hint when the input looks like an asset path, and can automatically retry via `spawn_object_from_path()`.

---

