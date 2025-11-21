import pathlib
import gradio as gr
import avtrans

import sys
import argparse
from omegaconf import OmegaConf

sys.path.append("LatentSync")
from scripts import inference

data_dir = pathlib.Path("generated")
data_dir.mkdir(exist_ok=True)


def create_args(
    checkpoint_path: str,
    video_path: str,
    audio_path: str,
    output_path: str,
    inference_steps: int,
    guidance_scale: float,
    seed: int,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inference_ckpt_path", type=str, required=True)
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--audio_path", type=str, required=True)
    parser.add_argument("--video_out_path", type=str, required=True)
    parser.add_argument("--inference_steps", type=int, default=20)
    parser.add_argument("--guidance_scale", type=float, default=1.5)
    parser.add_argument("--temp_dir", type=str, default="temp")
    parser.add_argument("--seed", type=int, default=1247)
    parser.add_argument("--enable_deepcache", action="store_true")

    return parser.parse_args(
        [
            "--inference_ckpt_path",
            checkpoint_path.absolute().as_posix(),
            "--video_path",
            pathlib.Path(video_path).absolute().as_posix(),
            "--audio_path",
            pathlib.Path(audio_path).absolute().as_posix(),
            "--video_out_path",
            pathlib.Path(output_path).absolute().as_posix(),
            "--inference_steps",
            str(inference_steps),
            "--guidance_scale",
            str(guidance_scale),
            "--seed",
            str(seed),
            "--temp_dir",
            "temp",
            "--enable_deepcache",
        ]
    )


def run_lipsync(
    video_path,
    audio_path,
    vid_output_path,
    guidance_scale=1.5,
    inference_steps=20,
):
    config_path = pathlib.Path("LatentSync/configs/unet/stage2_512.yaml")
    checkpoint_path = pathlib.Path("LatentSyncCkpts/latentsync_unet.pt")

    config = OmegaConf.load(config_path)

    config["run"].update(
        {
            "guidance_scale": guidance_scale,
            "inference_steps": inference_steps,
        }
    )
    args = create_args(
        checkpoint_path,
        video_path,
        audio_path,
        vid_output_path,
        inference_steps,
        guidance_scale,
        42,
    )

    result = inference.main(
        config=config,
        args=args,
    )

    return vid_output_path


def translate_video(
    video_file,
    srt_path,
    use_lipsync,
    enhance_audio,
    tts_model,
    target_language,
):
    video_file = pathlib.Path(video_file)
    srt_path = pathlib.Path(srt_path)
    stem = video_file.stem

    if tts_model == "xtts_v2":
        voice_clone_func = avtrans.voice_clone.xtts2_generate_from_transcript
    elif tts_model == "tacotron2":
        voice_clone_func = avtrans.voice_clone.tacotron2_generate_from_transcript

    final_video_path, translated_audio_path, abs_duration_diff = (
        avtrans.main.translate_audio(
            video_path=video_file,
            srt_path=srt_path,
            extracted_audio_path=data_dir
            / f"{stem}_extracted_audio_{target_language}_{tts_model}.wav",
            translated_audio_path=data_dir
            / f"{stem}_translated_audio_{target_language}_{tts_model}.wav",
            translated_text_path=data_dir
            / f"{stem}_translated_text_{target_language}_{tts_model}.txt",
            final_video_path=data_dir
            / f"{stem}_translated_video_{target_language}_{tts_model}.mp4",
            chunks_path=data_dir / f"{stem}_chunks_{target_language}_{tts_model}",
            target_language=target_language,
            enhance_audio=enhance_audio,
            voice_clone_func=voice_clone_func,
            translate_func=avtrans.translate.helsinki_translate,
        )
    )

    if use_lipsync:
        lipsync_vid_path = (
            data_dir
            / f"{stem}_translated_video_{target_language}_{tts_model}_lipsync.mp4"
        )

        run_lipsync(
            video_path=lipsync_vid_path,
            audio_path=translated_audio_path,
            vid_output_path=final_video_path,
            guidance_scale=1.5,
            inference_steps=20,
        )

        return lipsync_vid_path, translated_audio_path, round(abs_duration_diff, 2)
    else:
        return final_video_path, translated_audio_path, round(abs_duration_diff, 2)


demo = gr.Interface(
    fn=translate_video,
    inputs=[
        gr.Video(label="Upload Video", value="sample_data/Tanzania-2.mp4"),
        gr.File(
            label="Upload SRT",
            file_count="single",
            file_types=[".srt"],
            value="sample_data/Tanzania-caption.srt",
        ),
        gr.Checkbox(label="Add Lipsync", value=True),
        gr.Checkbox(label="Enhance translated Audio", value=True),
        gr.Dropdown(choices=["xtts_v2", "tacotron2"], label="TTS model to use"),
        gr.Dropdown(choices=["de"], label="Target Language"),
    ],
    outputs=[
        gr.Video(label="Translated Video"),
        gr.Audio(label="Translated Audio", type="filepath"),
        gr.Textbox(label="Absolute Duration Difference"),
    ],
    title='English to German Video Translator, built by <a href="https://sambhav300899.github.io/">Sambhav Rohatgi</a>',
    description="Upload a video and select a target language to translate the audio.",
)

if __name__ == "__main__":
    demo.launch()
