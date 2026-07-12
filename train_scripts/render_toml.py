#!/usr/bin/env python
"""
Render a musubi-tuner dataset TOML from CLI flags. Stdlib only (no deps).

Used by the generic cache/train .bat drivers so a single set of scripts can
drive any subject without hand-editing per-subject .toml files.

Architectures:
  krea2  : single image dataset
  klein  : single image dataset
  zimage : image dataset + optional --extra-dir image slot
  ltx    : 3 optional slots (video/image/extra), frame_sample emitted
  wan22  : 3 optional slots (video/image/extra), frame_sample omitted by default

Image slots enabled by --dataset (main) and --extra-dir (optional).
Video slots enabled by passing --video-dir / --image-dir / --extra-dir.

Cache directories are derived automatically from --cache-root + --name:
  image arches : <cache-root>/<name>_<arch>/{img,extra}
  video arches : <cache-root>/<name>_<arch>/{v,img,extra}

Example:
  python render_toml.py --arch krea2 --name courtney --out gen/courtney_krea2.toml \
      --cache-root D:/github/musubi-tuner/cache \
      --dataset D:/DATA/training/datasets/courtney/img --repeats 2 --res 256x384
"""

import argparse
import sys
from pathlib import Path

IMAGE_ARCHES = {"krea2", "klein", "zimage"}
VIDEO_ARCHES = {"ltx", "wan22"}


def fwd(p: str) -> str:
    """Normalize a path to forward slashes for TOML."""
    return str(p).replace("\\", "/")


def parse_res(s: str):
    parts = [int(x) for x in s.replace("x", ",").replace(" ", "").split(",") if x != ""]
    if len(parts) == 1:
        parts = [parts[0], parts[0]]
    if len(parts) != 2:
        sys.exit(f"ERROR: --res must be 'WxH' or 'W,H' (got {s!r})")
    return parts


def parse_int_list(s: str):
    return [int(x) for x in s.replace(" ", "").split(",") if x != ""]


def general_block(res, caption_ext, batch_size) -> str:
    w, h = res
    return (
        "[general]\n"
        f"resolution = [{w}, {h}]\n"
        f'caption_extension = "{caption_ext}"\n'
        f"batch_size = {batch_size}\n"
        "enable_bucket = true\n"
        "bucket_no_upscale = false\n"
    )


def image_dataset(image_dir, cache_dir, repeats) -> str:
    return (
        "\n[[datasets]]\n"
        f'image_directory = "{fwd(image_dir)}"\n'
        f'cache_directory = "{fwd(cache_dir)}"\n'
        f"num_repeats = {repeats}\n"
    )


def video_dataset(video_dir, cache_dir, repeats, frames, extraction, sample) -> str:
    block = (
        "\n[[datasets]]\n"
        f'video_directory = "{fwd(video_dir)}"\n'
        f'cache_directory = "{fwd(cache_dir)}"\n'
        f"num_repeats = {repeats}\n"
        f"target_frames = [{', '.join(str(f) for f in frames)}]\n"
        f'frame_extraction = "{extraction}"\n'
    )
    if str(sample).lower() not in ("", "none", "0"):
        block += f"frame_sample = {int(sample)}\n"
    return block


def build_image(args) -> str:
    if not args.dataset:
        sys.exit(f"ERROR: --dataset is required for arch {args.arch}")
    res = parse_res(args.res)
    base = f"{fwd(args.cache_root)}/{args.name}_{args.arch}"
    out = general_block(res, args.caption_ext, args.batch_size)
    out += image_dataset(args.dataset, f"{base}/img", args.repeats)
    if args.extra_dir:
        out += image_dataset(args.extra_dir, f"{base}/extra", args.extra_repeats)
    return out


def build_video(args) -> str:
    res = parse_res(args.res)
    base = f"{fwd(args.cache_root)}/{args.name}_{args.arch}"
    out = general_block(res, args.caption_ext, args.batch_size)

    blocks = 0
    if args.video_dir:
        out += video_dataset(
            args.video_dir,
            f"{base}/v",
            args.video_repeats,
            parse_int_list(args.video_frames),
            args.frame_extraction,
            args.frame_sample,
        )
        blocks += 1
    if args.image_dir:
        out += image_dataset(args.image_dir, f"{base}/img", args.image_repeats)
        blocks += 1
    if args.extra_dir:
        out += image_dataset(args.extra_dir, f"{base}/extra", args.extra_repeats)
        blocks += 1

    if blocks == 0:
        sys.exit(
            "ERROR: video arch requires at least one of "
            "--video-dir / --image-dir / --extra-dir"
        )
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arch", required=True, choices=sorted(IMAGE_ARCHES | VIDEO_ARCHES))
    p.add_argument("--name", required=True)
    p.add_argument("--out", required=True, help="TOML file to write.")
    p.add_argument("--cache-root", required=True, help="Base cache directory.")
    p.add_argument("--res", default="512x768", help="Resolution 'WxH' or 'W,H'.")
    p.add_argument("--caption-ext", default=".txt")
    p.add_argument("--batch-size", type=int, default=1)

    # image slot (main) — krea2 / klein / zimage
    p.add_argument("--dataset", default="", help="Main image directory.")
    p.add_argument("--repeats", type=int, default=2)

    # video slot — ltx / wan22
    p.add_argument("--video-dir", default="")
    p.add_argument("--video-repeats", type=int, default=4)
    p.add_argument("--video-frames", default="1,9,17,25")
    p.add_argument("--frame-extraction", default="uniform")
    p.add_argument("--frame-sample", default="4", help="Set 'none' to omit the line.")
    # image slot — ltx / wan22
    p.add_argument("--image-dir", default="")
    p.add_argument("--image-repeats", type=int, default=2)
    # extra image slot — zimage / ltx / wan22 (e.g. a nudes/close-up set)
    p.add_argument("--extra-dir", default="")
    p.add_argument("--extra-repeats", type=int, default=4)

    args = p.parse_args()

    if args.arch in IMAGE_ARCHES:
        content = build_image(args)
    else:
        content = build_video(args)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(content, encoding="utf-8")
    print(f"[render_toml] wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
