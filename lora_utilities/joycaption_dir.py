#!/usr/bin/env python
"""
JoyCaption batch captioner for LoRA dataset prep.

Captions every image in a directory with JoyCaption Beta One and writes a
`.txt` sidecar next to each image (ImageName.png -> ImageName.txt), with a
trigger word prepended for LoRA training.

Usage:
    python joycaption_dir.py <dir> <trigger>
    python joycaption_dir.py ./dataset underbust_corset
    python joycaption_dir.py ./dataset underbust_corset --style short --overwrite

Requires (once):
    pip install torch transformers accelerate pillow
The model (~17GB) auto-downloads from HuggingFace on first run.
"""

import argparse
import sys
import time
from pathlib import Path

import torch
from PIL import Image, ImageOps
from transformers import AutoProcessor, LlavaForConditionalGeneration

MODEL_ID = "fancyfeast/llama-joycaption-beta-one-hf-llava"

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tiff", ".jfif"}

# Prompt presets. Override any of these with --prompt "..."
PROMPTS = {
    "descriptive": "Write a detailed description for this image.",
    "short": "Write a short, one-paragraph description for this image.",
    "training": (
        "Write a descriptive caption for this image in a formal tone. "
        "Describe the subject, clothing, pose, and setting. "
        "Do not mention the image's resolution or that it is an image."
    ),
    "tags": (
        "Write a comma-separated list of booru-style tags describing this "
        "image, including subject, clothing, pose, and background."
    ),
}

SYSTEM_PROMPT = "You are a helpful image captioner."


def parse_args():
    p = argparse.ArgumentParser(
        description="Batch-caption a directory of images with JoyCaption Beta One.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("directory", help="Directory of images to caption.")
    p.add_argument(
        "trigger",
        help="Trigger word/phrase prepended to every caption (LoRA activation token).",
    )
    p.add_argument(
        "--style",
        choices=list(PROMPTS),
        default="training",
        help="Caption style preset.",
    )
    p.add_argument(
        "--prompt",
        default=None,
        help="Custom caption prompt (overrides --style).",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Recaption images that already have a .txt sidecar.",
    )
    p.add_argument(
        "--recursive",
        action="store_true",
        help="Also caption images in subdirectories.",
    )
    p.add_argument(
        "--no-trigger-comma",
        action="store_true",
        help="Join trigger and caption with a space instead of ', '.",
    )
    p.add_argument("--max-new-tokens", type=int, default=300)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument(
        "--greedy",
        action="store_true",
        help="Deterministic decoding (ignores temperature/top-p).",
    )
    p.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device.",
    )
    return p.parse_args()


def find_images(directory: Path, recursive: bool):
    it = directory.rglob("*") if recursive else directory.iterdir()
    return sorted(p for p in it if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def load_model(device: str):
    print(f"Loading {MODEL_ID} on {device} ...", flush=True)
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        device_map=device if device.startswith("cuda") else None,
    )
    if not device.startswith("cuda"):
        model = model.to(device)
    model.eval()
    return processor, model, dtype


@torch.no_grad()
def caption_image(image: Image.Image, prompt: str, processor, model, dtype, args) -> str:
    convo = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    convo_string = processor.apply_chat_template(
        convo, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(text=[convo_string], images=[image], return_tensors="pt").to(
        model.device
    )
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(dtype)

    gen_kwargs = dict(max_new_tokens=args.max_new_tokens, use_cache=True)
    if args.greedy:
        gen_kwargs.update(do_sample=False)
    else:
        gen_kwargs.update(
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
        )

    generate_ids = model.generate(**inputs, **gen_kwargs)[0]
    generate_ids = generate_ids[inputs["input_ids"].shape[1]:]
    caption = processor.tokenizer.decode(
        generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return caption.strip()


def main():
    args = parse_args()

    directory = Path(args.directory).expanduser()
    if not directory.is_dir():
        sys.exit(f"Error: not a directory: {directory}")

    prompt = args.prompt or PROMPTS[args.style]
    joiner = " " if args.no_trigger_comma else ", "

    images = find_images(directory, args.recursive)
    if not images:
        sys.exit(f"No images found in {directory}")

    print(f"Found {len(images)} image(s). Trigger: '{args.trigger}'")

    processor, model, dtype = load_model(args.device)

    done = skipped = failed = 0
    t0 = time.time()

    for i, img_path in enumerate(images, 1):
        txt_path = img_path.with_suffix(".txt")
        if txt_path.exists() and not args.overwrite:
            skipped += 1
            print(f"[{i}/{len(images)}] skip (exists): {img_path.name}")
            continue

        try:
            image = Image.open(img_path)
            image = ImageOps.exif_transpose(image).convert("RGB")
        except Exception as e:
            failed += 1
            print(f"[{i}/{len(images)}] FAILED to open {img_path.name}: {e}")
            continue

        try:
            caption = caption_image(image, prompt, processor, model, dtype, args)
        except Exception as e:
            failed += 1
            print(f"[{i}/{len(images)}] FAILED to caption {img_path.name}: {e}")
            continue

        final = f"{args.trigger}{joiner}{caption}" if args.trigger else caption
        txt_path.write_text(final, encoding="utf-8")
        done += 1
        preview = final if len(final) <= 100 else final[:97] + "..."
        print(f"[{i}/{len(images)}] {img_path.name} -> {preview}")

    dt = time.time() - t0
    print(
        f"\nDone. captioned={done} skipped={skipped} failed={failed} "
        f"in {dt:.0f}s ({dt / max(done, 1):.1f}s/img)"
    )


if __name__ == "__main__":
    main()
