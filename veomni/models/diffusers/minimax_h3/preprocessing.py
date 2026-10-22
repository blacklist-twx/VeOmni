"""Prepare raw Ref2VA examples for the existing ``dit_offline`` training path."""

from __future__ import annotations

import math
from pathlib import Path

import torch
from PIL import Image

from .inference import BasePipeline, MiniMaxH3Unit_ReferenceEncoder
from .minimax_h3_core.minimax_h3_text_encoder import encode_prompt
from .minimax_h3_core.packed_sequence import build_packed_ref2va


def _cpu(value):
    if torch.is_tensor(value):
        return value.detach().cpu().contiguous()
    if isinstance(value, dict):
        return {key: _cpu(item) for key, item in value.items()}
    return value


def _aligned_audio(audio, frames):
    waveform, rate = audio
    waveform = torch.as_tensor(waveform)
    if waveform.ndim != 2 or waveform.shape[0] not in (1, 2) or waveform.shape[1] == 0 or rate <= 0:
        raise ValueError("Audio requires a nonempty mono/stereo [C,T] waveform and positive sample rate.")
    samples = int(frames / 24 * rate)
    waveform = waveform[..., :samples]
    return torch.nn.functional.pad(waveform, (0, max(0, samples - waveform.shape[-1]))), rate


@torch.no_grad()
def prepare_ref2va_sample(
    condition,
    *,
    prompt: str,
    video: list[Image.Image],
    references: list[dict],
    audio: tuple[torch.Tensor, int] | None = None,
    device="cuda",
    seed: int = 42,
    ref_image_short_edge: int = 512,
    ref_video_short_edge: int = 480,
    ref_video_max_pixels: int = 480 * 832,
    negative_prompt: str | None = None,
) -> dict:
    """Encode one decoded example, staging frozen encoders without loading a DiT.

    Target frames must share a 32-aligned size and use 24 fps / 17n+5 frames.
    References use native inference dictionaries (PIL images, 24 fps frame
    lists, or [C,T] waveforms plus sample_rate), in presentation order. Native
    reference resizing preserves aspect ratio up to 32-pixel alignment.
    Video soundtracks and target audio are trimmed/padded to the selected clip;
    standalone audio references retain their original duration. Output is a
    decoded CPU record for OfflineEmbeddingSaver / dit_offline.
    """
    if not isinstance(prompt, str) or not prompt:
        raise ValueError("prompt must be a nonempty string.")
    if len(video) < 22 or len(video) % 17 != 5:
        raise ValueError("Target video must have 17n+5 frames with n >= 1 at 24 fps.")
    width, height = video[0].size
    if width % 32 or height % 32 or any(frame.size != (width, height) for frame in video):
        raise ValueError("Target frames must share a size divisible by 32.")
    if not references:
        raise ValueError("Ref2VA requires at least one reference.")
    if min(ref_image_short_edge, ref_video_short_edge, ref_video_max_pixels) <= 0:
        raise ValueError("Reference resize limits must be positive.")
    if any(getattr(condition, key) is None for key in ("_video_vae", "_audio_vae", "_text_encoder", "_tokenizer")):
        raise ValueError("Load the condition encoders and processor before preparing Ref2VA caches.")

    # Share the native inference preprocessing and staging machinery; no DiT is constructed.
    pipe = BasePipeline(device=device, torch_dtype=torch.bfloat16)
    pipe.video_vae = condition._video_vae
    pipe.audio_vae = condition._audio_vae
    pipe.text_encoder = condition._text_encoder
    pipe.imgvid_cond_noise_aug = condition.config.imgvid_cond_noise_aug
    pipe.audio_cond_noise_aug = condition.config.audio_cond_noise_aug
    encoder = MiniMaxH3Unit_ReferenceEncoder()
    refs = []
    for original in references:
        ref = dict(original)
        kind = ref.get("type")
        if kind not in ("image", "video", "video_audio", "audio"):
            raise ValueError(f"Unknown reference type: {kind!r}")
        if kind in ("video", "video_audio"):
            frames = ref.get("video", [])
            if len(frames) < 22:
                raise ValueError("Video references require at least 22 frames at 24 fps.")
            count = encoder._trim_reference_video_length(pipe, min(len(frames), len(video)))
            ref["video"] = frames[:count]
            if kind == "video_audio":
                ref["audio"], ref["sample_rate"] = _aligned_audio((ref["audio"], ref["sample_rate"]), count)
        elif kind == "audio":
            waveform = torch.as_tensor(ref["audio"])
            if (
                waveform.ndim != 2
                or waveform.shape[0] not in (1, 2)
                or waveform.shape[1] == 0
                or ref["sample_rate"] <= 0
            ):
                raise ValueError("Audio references require nonempty mono/stereo [C,T] and positive sample_rate.")
            ref["audio"] = waveform
        refs.append(ref)

    pipe.load_models_to_device(["video_vae"])
    video_tensor = pipe.preprocess_video(
        [frame.convert("RGB") for frame in video], torch_dtype=torch.float32, min_value=0
    )
    video_latent = condition._encode_video(video_tensor, torch.device(device))
    del video_tensor
    if audio is None:
        audio_latent = condition._make_silent_audio_latent(len(video), torch.device(device))
    else:
        pipe.load_models_to_device(["audio_vae"])
        audio_latent = condition._encode_audio(_aligned_audio(audio, len(video)), len(video), torch.device(device))
    reference = encoder.process(
        pipe, refs, seed, video_latent, len(video), ref_image_short_edge, ref_video_short_edge, ref_video_max_pixels
    )
    pipe.load_models_to_device(["text_encoder"])
    encoded = encode_prompt(
        pipe.text_encoder,
        condition._processor,
        condition._tokenizer,
        prompt,
        device=device,
        dtype=pipe.torch_dtype,
        ref_blocks=reference["ref_blocks"],
    )
    packed = build_packed_ref2va(
        text_len=encoded["prompt_embeds"].shape[0],
        latent_t=video_latent.shape[2],
        latent_h=video_latent.shape[3],
        latent_w=video_latent.shape[4],
        audio_t=audio_latent.shape[-1],
        audio_channel=audio_latent.shape[0],
        ref_blocks=reference["ref_blocks"],
        text_token_tags=encoded["text_token_tags"],
    )
    sample = _cpu(
        dict(
            input_latents=video_latent,
            audio_input_latents=audio_latent,
            prompt_embeds=encoded["prompt_embeds"],
            packed=packed,
            ref_visual_anchor=reference["ref_visual_anchor"],
            ref_audio_anchor=reference["ref_audio_anchor"],
            imgvid_cond_noise_aug=pipe.imgvid_cond_noise_aug,
            audio_cond_noise_aug=pipe.audio_cond_noise_aug,
            has_audio=audio is not None,
        )
    )
    if negative_prompt is not None:
        sample = condition.add_unconditional_cache(
            sample, negative_prompt=negative_prompt, ref_blocks=reference["ref_blocks"]
        )
    return sample


def read_video(path: str | Path, max_frames: int) -> list[Image.Image]:
    """Sequentially sample a video at 24 fps, stopping once enough frames exist."""
    import imageio

    frames = []
    reader = imageio.get_reader(str(path))
    try:
        fps = float(reader.get_meta_data()["fps"])
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError(f"Invalid video frame rate: {fps}")
        desired = [round(i / 24 * fps) for i in range(max_frames)]
        for index, frame in enumerate(reader.iter_data()):
            while len(frames) < max_frames and desired[len(frames)] <= index:
                frames.append(Image.fromarray(frame).convert("RGB"))
            if len(frames) == max_frames:
                break
    finally:
        reader.close()
    count = ((len(frames) - 5) // 17) * 17 + 5
    if count < 22:
        raise ValueError(f"Video {path} has fewer than 22 usable frames at 24 fps.")
    return frames[:count]


def load_ref2va_record(record: dict, *, root: Path, num_frames=73, height=480, width=832) -> dict:
    """Decode one JSONL record; relative paths are resolved against its manifest."""
    import torchaudio
    import torchvision.transforms.functional as TF

    if num_frames < 22 or num_frames % 17 != 5 or min(height, width) <= 0 or height % 32 or width % 32:
        raise ValueError("Use 17n+5 frames (n >= 1) and positive 32-aligned height/width.")

    def read_audio(path):
        return torchaudio.load(str(root / path))

    target = read_video(root / record["video"], num_frames)
    resized = []
    for frame in target:
        scale = max(width / frame.width, height / frame.height)
        frame = TF.resize(
            frame, (round(frame.height * scale), round(frame.width * scale)), TF.InterpolationMode.BILINEAR
        )
        resized.append(TF.center_crop(frame, (height, width)))
    refs = []
    for item in record["references"]:
        kind = item["type"]
        ref = {"type": kind}
        if kind == "image":
            with Image.open(root / item["image"]) as image:
                ref["image"] = image.convert("RGB")
        elif kind in ("video", "video_audio"):
            ref["video"] = read_video(root / item["video"], len(resized))
            if kind == "video" and item.get("audio") is not None:
                raise ValueError("Use video_audio to supply a reference soundtrack.")
        elif kind != "audio":
            raise ValueError(f"Unknown reference type: {kind!r}")
        if kind in ("audio", "video_audio"):
            ref["audio"], ref["sample_rate"] = read_audio(item["audio"])
        refs.append(ref)
    return dict(
        prompt=record["prompt"],
        video=resized,
        audio=read_audio(record["audio"]) if record.get("audio") else None,
        references=refs,
    )
