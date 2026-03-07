import argparse
import json
import shutil
import tempfile
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from result import Do, Err, Ok, Result, do_notation

# --- 1. Data Structures ---
type Theme = dict[str, Any]
type Manifest = dict[str, Any]


@dataclass(frozen=True)
class CustomIcon:
    name: str
    source_path: Path
    suffixes: list[str]
    stems: list[str]


# --- 2. Categorized Errors ---
class PatcherError(StrEnum):
    """Categorized error states for patcher operations."""

    PATH_MISSING = "path_missing"
    SOURCE_NOT_FOUND = "source_not_found"
    IO_INJECTION_FAILED = "io_injection_failed"
    IO_CLEANUP_FAILED = "io_cleanup_failed"
    JSON_INVALID_STRUCTURE = "json_invalid_structure"
    JSON_DECODE_FAILED = "json_decode_failed"
    LOGIC_PATCHING_FAILED = "logic_patching_failed"
    LOGIC_CLEANING_FAILED = "logic_cleaning_failed"
    ATOMIC_WRITE_FAILURE = "atomic_write_failure"
    INTERNAL_ERROR = "internal_error"


# --- 3. Pure Logic Transformations (Sub-helpers) ---
def _identify_flavor(theme_name: str) -> str:
    """Returns the flavor name based on the theme name."""
    if "Latte" in theme_name:
        return "latte"
    if any(f in theme_name for f in ["Frappé", "Frappe"]):
        return "frappe"
    if "Macchiato" in theme_name:
        return "macchiato"
    if "Mocha" in theme_name:
        return "mocha"
    return ""


def _patch_single_theme(
    theme: Theme,
    current_flavor: str,
    custom_icons: list[CustomIcon],
    target_suffixes: dict[str, str],
    target_stems: dict[str, str],
) -> int:
    """Mutates a single theme dict and returns the number of changes made."""
    patch_count = 0

    # 1. Register SVGs
    file_icons: dict[str, dict[str, str]] = theme.setdefault("file_icons", {})
    for icon in custom_icons:
        expected_path = f"./icons/{current_flavor}/{icon.name}.svg"
        if file_icons.get(icon.name, {}).get("path") != expected_path:
            file_icons[icon.name] = {"path": expected_path}
            patch_count += 1

        for suffix in icon.suffixes:
            target_suffixes[suffix] = icon.name
        for stem in icon.stems:
            target_stems[stem] = icon.name

    # 2. Patch Suffixes
    suffixes: dict[str, str] = theme.setdefault("file_suffixes", {})
    for ext, icon_key in target_suffixes.items():
        if suffixes.get(ext) != icon_key:
            suffixes[ext] = icon_key
            patch_count += 1

    # 3. Patch Stems
    stems: dict[str, str] = theme.setdefault("file_stems", {})
    for stem, icon_key in target_stems.items():
        if stems.get(stem) != icon_key:
            stems[stem] = icon_key
            patch_count += 1

    return patch_count


def _clean_single_theme(
    theme: Theme,
    custom_icons: list[CustomIcon],
    keys_suffixes: list[str],
    keys_stems: list[str],
    *,
    clean_icons: bool,
    clean_aliases: bool,
) -> int:
    """Mutates a single theme dict to remove injected state."""
    mutations = 0
    if clean_icons:
        file_icons: dict[str, Any] = theme.get("file_icons", {})
        for icon in custom_icons:
            if icon.name in file_icons:
                del file_icons[icon.name]
                mutations += 1

    if clean_aliases:
        suffixes: dict[str, str] = theme.get("file_suffixes", {})
        for ext in keys_suffixes:
            if ext in suffixes:
                del suffixes[ext]
                mutations += 1

        stems: dict[str, str] = theme.get("file_stems", {})
        for stem in keys_stems:
            if stem in stems:
                del stems[stem]
                mutations += 1
    return mutations


# --- 4. Pure Logic Transformations ---
def calculate_patched_manifest(
    data: Manifest, custom_icons: list[CustomIcon], target_suffixes: dict[str, str], target_stems: dict[str, str]
) -> Result[tuple[Manifest, int], PatcherError]:
    """Pure transformation that patches the manifest data. Returns (new_data, patch_count)."""
    try:
        themes: list[Theme] = data.get("themes", [])
        if not themes:
            return Err(PatcherError.JSON_INVALID_STRUCTURE)

        total_patches = 0
        for theme in themes:
            flavor = _identify_flavor(theme.get("name", ""))
            if flavor:
                total_patches += _patch_single_theme(theme, flavor, custom_icons, target_suffixes, target_stems)

        return Ok((data, total_patches))
    except KeyError, AttributeError, TypeError:
        return Err(PatcherError.LOGIC_PATCHING_FAILED)


def calculate_cleaned_manifest(
    data: Manifest,
    custom_icons: list[CustomIcon],
    keys_to_remove_suffixes: list[str],
    keys_to_remove_stems: list[str],
    *,
    clean_icons: bool,
    clean_aliases: bool,
) -> Result[tuple[Manifest, int], PatcherError]:
    """Pure transformation that cleans the manifest data. Returns (new_data, mutations)."""
    try:
        themes: list[Theme] = data.get("themes", [])
        if not themes:
            return Err(PatcherError.JSON_INVALID_STRUCTURE)

        mutations = 0
        for theme in themes:
            mutations += _clean_single_theme(
                theme,
                custom_icons,
                keys_to_remove_suffixes,
                keys_to_remove_stems,
                clean_icons=clean_icons,
                clean_aliases=clean_aliases,
            )

        return Ok((data, mutations))
    except KeyError, AttributeError, TypeError:
        return Err(PatcherError.LOGIC_CLEANING_FAILED)


# --- 5. The Patcher System (IO Orchestration) ---
def read_json(path: Path) -> Result[Manifest, PatcherError]:
    """Reads and parses a JSON file into a Result."""
    try:
        with path.open("r", encoding="utf-8") as f:
            data: Manifest = json.load(f)
            return Ok(data)
    except OSError, json.JSONDecodeError:
        return Err(PatcherError.JSON_DECODE_FAILED)


def _inject_svg_files(
    icons_dir: Path, custom_icons: list[CustomIcon], flavors: list[str]
) -> Result[None, PatcherError]:
    """Copies SVG files to destination directories."""
    try:
        for icon in custom_icons:
            for flavor in flavors:
                dest_dir = icons_dir / flavor
                dest_file = dest_dir / f"{icon.name}.svg"
                shutil.copy2(icon.source_path, dest_file)
        return Ok(None)
    except OSError:
        return Err(PatcherError.IO_INJECTION_FAILED)


def _delete_svg_files(icons_dir: Path, custom_icons: list[CustomIcon], flavors: list[str]) -> Result[int, PatcherError]:
    """Removes injected SVG files."""
    deleted_files = 0
    try:
        for icon in custom_icons:
            for flavor in flavors:
                svg_file = icons_dir / flavor / f"{icon.name}.svg"
                if svg_file.exists():
                    svg_file.unlink()
                    deleted_files += 1
        return Ok(deleted_files)
    except OSError:
        return Err(PatcherError.IO_CLEANUP_FAILED)


def inject_and_patch_theme(
    ext_base_dir: Path, custom_icons: list[CustomIcon], target_suffixes: dict[str, str], target_stems: dict[str, str]
) -> Result[str, PatcherError]:
    """Copies new SVGs into the theme directories and atomically patches the JSON manifest."""
    json_path = ext_base_dir / "icon_themes" / "catppuccin-icons.json"
    icons_dir = ext_base_dir / "icons"
    flavors = ["frappe", "latte", "macchiato", "mocha"]

    if not json_path.exists() or not icons_dir.exists():
        return Err(PatcherError.PATH_MISSING)

    if any(not icon.source_path.exists() for icon in custom_icons):
        return Err(PatcherError.SOURCE_NOT_FOUND)

    return (
        _inject_svg_files(icons_dir, custom_icons, flavors)
        .and_then(lambda _: read_json(json_path))
        .and_then(lambda data: calculate_patched_manifest(data, custom_icons, target_suffixes, target_stems))
        .and_then(
            lambda payload: (
                Ok(f"Success! {len(custom_icons)} SVGs copied, JSON was already up-to-date.")
                if payload[1] == 0
                else write_json_atomically(json_path, payload[0]).map(
                    lambda _: f"Success! Registered SVGs and applied {payload[1]} mapping updates."
                )
            )
        )
    )


@do_notation
def clean_theme_safely(
    ext_base_dir: Path,
    custom_icons: list[CustomIcon],
    target_suffixes: dict[str, str],
    target_stems: dict[str, str],
    *,
    clean_icons: bool,
    clean_aliases: bool,
) -> Do[str, PatcherError]:
    """Atomically strips injected SVGs and JSON aliases out of the theme."""
    json_path = ext_base_dir / "icon_themes" / "catppuccin-icons.json"
    icons_dir = ext_base_dir / "icons"
    flavors = ["frappe", "latte", "macchiato", "mocha"]

    if not json_path.exists():
        yield Err(PatcherError.PATH_MISSING)

    # 1. Handle SVG deletion (yield unwraps if Ok, short-circuits if Err)
    deleted_files = yield _delete_svg_files(icons_dir, custom_icons, flavors) if clean_icons else Ok(0)

    # 2. Load Manifest
    data = yield read_json(json_path)

    # 3. Calculate Clean State
    keys_suffixes = list(target_suffixes.keys()) + [s for icon in custom_icons for s in icon.suffixes]
    keys_stems = list(target_stems.keys()) + [s for icon in custom_icons for s in icon.stems]

    new_data, mutations = yield calculate_cleaned_manifest(
        data, custom_icons, keys_suffixes, keys_stems, clean_icons=clean_icons, clean_aliases=clean_aliases
    )

    # 4. Finish or Write
    if mutations == 0 and deleted_files == 0:
        return "Theme is already perfectly clean. No changes made."

    yield write_json_atomically(json_path, new_data)

    # Construct status message
    status: list[str] = []
    if clean_icons:
        status.append(f"Deleted {deleted_files} SVGs")
    if clean_aliases or clean_icons:
        status.append(f"Removed {mutations} JSON bindings")

    return f"Successfully cleaned state! ({', '.join(status)})"  # noqa: B901


def write_json_atomically(path: Path, data: Manifest) -> Result[None, PatcherError]:
    """Writes JSON data to a path atomically using a temporary file."""
    _fd, temp_path_str = tempfile.mkstemp(suffix=".json", text=True)
    temp_path = Path(temp_path_str)
    try:
        with temp_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        shutil.move(temp_path, path)
        return Ok(None)
    except OSError:
        if temp_path.exists():
            temp_path.unlink()
        return Err(PatcherError.ATOMIC_WRITE_FAILURE)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Zed Catppuccin Icon Patcher")
    parser.add_argument(
        "--clean-icon-state", action="store_true", help="Remove injected SVGs and unregister them from the JSON"
    )
    parser.add_argument(
        "--clean-alias-state", action="store_true", help="Remove custom file extensions and filename aliases"
    )
    args = parser.parse_args()

    catppuccin_base = Path.home() / "Library/Application Support/Zed/extensions/installed/catppuccin-icons"

    mappings_suffixes = {
        "ss": "scheme",
        "sbt": "scala",
    }

    mappings_stems = {
        ".ocamlformat": "ocaml",
        ".zanuda": "ocaml",
        "dune-project": "ocaml",
        "Gemfile": "ruby",
        "odinfmt.json": "odin",
        "dscanner.ini": "d",
        ".clang-format": "c",
        ".clang-tidy": "c",
        ".clangd": "c",
        "omnisharp.json": "csharp",
    }

    new_svgs = [
        CustomIcon(
            name="flix",
            source_path=Path("icon-export/final/flix-v3.svg").resolve(),
            suffixes=["flix"],
            stems=["flix.toml"],
        ),
        CustomIcon(name="pony", source_path=Path("icon-export/final/pony.svg").resolve(), suffixes=["pony"], stems=[]),
    ]

    # Route execution based on CLI flags
    if args.clean_icon_state or args.clean_alias_state:
        result = clean_theme_safely(
            catppuccin_base,
            new_svgs,
            mappings_suffixes,
            mappings_stems,
            clean_icons=args.clean_icon_state,
            clean_aliases=args.clean_alias_state,
        )
    else:
        result = inject_and_patch_theme(catppuccin_base, new_svgs, mappings_suffixes, mappings_stems)

    # Handle the Result explicitly
    match result:
        case Ok(str() as msg):
            print(f"Success: {msg}")
            print("Restart Zed (Cmd+Q) to load the changes.")
        case Err(PatcherError.PATH_MISSING):
            print("Operation Failed: Required theme paths are missing. Is the extension installed?")
        case Err(PatcherError.SOURCE_NOT_FOUND):
            print("Operation Failed: One or more source SVGs could not be found.")
        case Err(PatcherError.IO_INJECTION_FAILED):
            print("Operation Failed: System error while copying SVG files.")
        case Err(PatcherError.JSON_DECODE_FAILED):
            print("Operation Failed: Could not read the theme JSON manifest.")
        case Err(PatcherError() as other_error):
            print(f"Operation Failed: {other_error}")
        case Err(Exception() as other_exception):
            print(f"Operation Failed: {other_exception}")
        case _:
            print("Operation Failed: Unknown critical failure.")
