"""Stitch six camera visualizations and one LiDAR BEV into a video.

The default camera layout matches the custom dataset converter in this repo::

    camera-1 (front-left) | camera-0 (front) | camera-2 (front-right) | LiDAR
    camera-4 (rear-left)  | camera-3 (rear)  | camera-5 (rear-right)  | LiDAR

Both ``camera-0`` and ``camera0`` style directory names are accepted. Frames
are synchronized by filename stem, so only stems present in all seven input
directories are written.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from PIL import Image, ImageOps


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
DEFAULT_LAYOUT = (1, 0, 2, 4, 3, 5)

# Pillow < 9.1 exposes LANCZOS directly on Image, while newer releases put it
# under Image.Resampling. Keep the script usable in older training containers.
try:
    LANCZOS = Image.Resampling.LANCZOS
except AttributeError:
    LANCZOS = Image.LANCZOS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stitch 6 camera views + 1 LiDAR BEV and encode a video."
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Visualization root containing camera-0...camera-5 and lidar.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output video path (default: <input_dir>/detection_mosaic.mp4).",
    )
    parser.add_argument("--fps", type=float, default=10.0, help="Video FPS.")
    parser.add_argument(
        "--num-frames",
        type=int,
        default=None,
        help=(
            "Number of frames to write. By default all synchronized frames are "
            "used; for example, --num-frames 300."
        ),
    )
    parser.add_argument(
        "--sampling",
        choices=("even", "first"),
        default="even",
        help=(
            "How --num-frames selects frames: evenly across the full sequence "
            "or only from its beginning (default: even)."
        ),
    )
    parser.add_argument(
        "--cell-width", type=int, default=640, help="Width of each camera cell."
    )
    parser.add_argument(
        "--cell-height", type=int, default=360, help="Height of each camera cell."
    )
    parser.add_argument(
        "--bev-width",
        type=int,
        default=None,
        help="LiDAR panel width (default: 2 * cell-height).",
    )
    parser.add_argument(
        "--layout",
        type=int,
        nargs=6,
        default=DEFAULT_LAYOUT,
        metavar=("TL", "TM", "TR", "BL", "BM", "BR"),
        help="Camera indices in top-left...bottom-right order.",
    )
    parser.add_argument(
        "--frames-dir",
        type=Path,
        default=None,
        help="Stitched-image directory (default: <input_dir>/stitched_frames).",
    )
    parser.add_argument(
        "--no-save-frames",
        action="store_true",
        help="Encode the video without saving the stitched PNG frames.",
    )
    parser.add_argument(
        "--ffmpeg",
        type=Path,
        default=None,
        help="FFmpeg executable; normally discovered automatically.",
    )
    parser.add_argument(
        "--codec",
        default="mpeg4",
        help="FFmpeg video codec (default: mpeg4 for broad compatibility).",
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=None,
        help=(
            "FFmpeg q:v quality from 1 (best) to 31 (worst). For mpeg4, "
            "the default is 2; omit it for other codecs."
        ),
    )
    parser.add_argument(
        "--crf",
        type=int,
        default=None,
        help="Optional H.264 quality (lower is better); omitted by default.",
    )
    parser.add_argument(
        "--preset",
        default=None,
        help="Optional FFmpeg encoder preset; omitted by default.",
    )
    args = parser.parse_args()

    if args.fps <= 0:
        parser.error("--fps must be positive")
    if args.num_frames is not None and args.num_frames <= 0:
        parser.error("--num-frames must be positive")
    if args.quality is not None and not 1 <= args.quality <= 31:
        parser.error("--quality must be between 1 and 31")
    if args.cell_width <= 0 or args.cell_height <= 0:
        parser.error("--cell-width and --cell-height must be positive")
    if args.bev_width is not None and args.bev_width <= 0:
        parser.error("--bev-width must be positive")
    if sorted(args.layout) != list(range(6)):
        parser.error("--layout must contain each camera index 0...5 exactly once")
    return args


def natural_key(value: str) -> List[object]:
    """Sort names containing numbers in human/numeric order."""
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", value)]


def resolve_camera_dir(root: Path, index: int) -> Path:
    candidates = (
        root / f"camera-{index}",
        root / f"camera{index}",
        root / f"camera_{index}",
    )
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    tried = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Camera {index} directory not found; tried: {tried}")


def collect_images(folder: Path) -> Dict[str, Path]:
    images: Dict[str, Path] = {}
    for path in folder.iterdir():
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        if path.stem in images:
            raise ValueError(
                f"Duplicate frame stem {path.stem!r} in {folder}; "
                "keep only one image extension for each frame."
            )
        images[path.stem] = path
    if not images:
        raise FileNotFoundError(f"No supported images found in {folder}")
    return images


def find_ffmpeg(explicit: Path | None) -> Path:
    candidates: List[Path] = []
    if explicit is not None:
        candidates.append(explicit)
    discovered = shutil.which("ffmpeg")
    if discovered:
        candidates.append(Path(discovered))
    # Common locations inside Windows and Linux/macOS Conda environments.
    candidates.extend(
        [
            Path(sys.prefix) / "Library" / "bin" / "ffmpeg.exe",
            Path(sys.prefix) / "bin" / "ffmpeg",
        ]
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        "FFmpeg was not found. Install ffmpeg or pass --ffmpeg /path/to/ffmpeg."
    )


def fit_image(path: Path, size: Sequence[int]) -> Image.Image:
    """Resize with aspect ratio preserved and pad unused space in black."""
    with Image.open(path) as source:
        image = source.convert("RGB")
        return ImageOps.pad(
            image,
            tuple(size),
            method=LANCZOS,
            color=(0, 0, 0),
            centering=(0.5, 0.5),
        )


def compose_frame(
    stem: str,
    camera_images: Sequence[Dict[str, Path]],
    lidar_images: Dict[str, Path],
    cell_width: int,
    cell_height: int,
    bev_width: int,
) -> Image.Image:
    camera_panel_width = 3 * cell_width
    frame_height = 2 * cell_height
    canvas = Image.new(
        "RGB", (camera_panel_width + bev_width, frame_height), (0, 0, 0)
    )

    for position, images in enumerate(camera_images):
        row, column = divmod(position, 3)
        tile = fit_image(images[stem], (cell_width, cell_height))
        canvas.paste(tile, (column * cell_width, row * cell_height))

    bev = fit_image(lidar_images[stem], (bev_width, frame_height))
    canvas.paste(bev, (camera_panel_width, 0))
    return canvas


def common_frame_stems(image_sets: Iterable[Dict[str, Path]]) -> List[str]:
    sets = [set(images) for images in image_sets]
    common = set.intersection(*sets)
    if not common:
        raise ValueError("No common frame filename stems exist across all 7 folders")
    return sorted(common, key=natural_key)


def select_frame_stems(
    stems: Sequence[str], num_frames: int | None, sampling: str
) -> List[str]:
    """Select at most num_frames, optionally distributed over the full clip."""
    if num_frames is None or num_frames >= len(stems):
        return list(stems)
    if sampling == "first":
        return list(stems[:num_frames])

    # Integer arithmetic gives deterministic, unique indices including both
    # endpoints. For 3000 -> 300, this samples the whole time span rather than
    # making a video from only the first tenth of the sequence.
    if num_frames == 1:
        indices = [0]
    else:
        last = len(stems) - 1
        indices = [
            (index * last) // (num_frames - 1) for index in range(num_frames)
        ]
    return [stems[index] for index in indices]


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    camera_dirs = [resolve_camera_dir(input_dir, index) for index in args.layout]
    lidar_dir = input_dir / "lidar"
    if not lidar_dir.is_dir():
        raise FileNotFoundError(f"LiDAR directory not found: {lidar_dir}")

    camera_images = [collect_images(folder) for folder in camera_dirs]
    lidar_images = collect_images(lidar_dir)
    all_images = [*camera_images, lidar_images]
    common_stems = common_frame_stems(all_images)

    counts = [len(images) for images in all_images]
    if any(count != len(common_stems) for count in counts):
        print(
            "Warning: folder frame counts are "
            f"{counts}; using their {len(common_stems)}-frame intersection.",
            file=sys.stderr,
        )
    stems = select_frame_stems(common_stems, args.num_frames, args.sampling)

    output = (args.output or (input_dir / "detection_mosaic.mp4")).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    frames_dir = (args.frames_dir or (input_dir / "stitched_frames")).resolve()
    if not args.no_save_frames:
        frames_dir.mkdir(parents=True, exist_ok=True)

    bev_width = args.bev_width or (2 * args.cell_height)
    frame_width = 3 * args.cell_width + bev_width
    frame_height = 2 * args.cell_height
    if frame_width % 2 or frame_height % 2:
        raise ValueError(
            f"Output size {frame_width}x{frame_height} is not even; "
            "H.264/yuv420p requires even width and height."
        )

    ffmpeg = find_ffmpeg(args.ffmpeg)
    command = [
        str(ffmpeg),
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pixel_format",
        "rgb24",
        "-video_size",
        f"{frame_width}x{frame_height}",
        "-framerate",
        str(args.fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        args.codec,
    ]
    if args.preset:
        command.extend(["-preset", args.preset])
    if args.crf is not None:
        command.extend(["-crf", str(args.crf)])
    quality = args.quality
    if quality is None and args.codec.lower() == "mpeg4":
        quality = 2
    if quality is not None:
        command.extend(["-q:v", str(quality)])
    command.extend(
        [
            "-pix_fmt",
            "yuv420p",
            str(output),
        ]
    )

    print(f"Input: {input_dir}")
    print(f"Layout camera indices: {list(args.layout[:3])} / {list(args.layout[3:])}")
    quality_text = f", q:v={quality}" if quality is not None else ""
    print(f"Codec: {args.codec}{quality_text}")
    selection = "all synchronized frames"
    if len(stems) < len(common_stems):
        selection = (
            f"{args.sampling} sampling from {len(common_stems)} synchronized frames"
        )
    print(
        f"Frames: {len(stems)} ({selection}), FPS: {args.fps:g}, "
        f"size: {frame_width}x{frame_height}"
    )

    creationflags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stderr=subprocess.PIPE,
        creationflags=creationflags,
    )
    assert process.stdin is not None
    try:
        for number, stem in enumerate(stems, start=1):
            frame = compose_frame(
                stem,
                camera_images,
                lidar_images,
                args.cell_width,
                args.cell_height,
                bev_width,
            )
            if not args.no_save_frames:
                frame.save(frames_dir / f"{stem}.png")
            process.stdin.write(frame.tobytes())
            print(f"\rEncoding frame {number}/{len(stems)}", end="", flush=True)
    except BrokenPipeError as error:
        stderr = process.stderr.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"FFmpeg stopped while encoding:\n{stderr}") from error
    finally:
        process.stdin.close()

    stderr = process.stderr.read().decode("utf-8", errors="replace")
    return_code = process.wait()
    print()
    if return_code != 0:
        raise RuntimeError(f"FFmpeg failed with exit code {return_code}:\n{stderr}")

    print(f"Video written: {output}")
    if not args.no_save_frames:
        print(f"Stitched frames written: {frames_dir}")


if __name__ == "__main__":
    main()
