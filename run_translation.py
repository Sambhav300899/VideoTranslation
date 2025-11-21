import torch
import argparse
from gradio_demo import translate_video
import ast

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Video Translation")
    parser.add_argument("--i", type=str, required=True)
    parser.add_argument("--o", type=str, required=True)
    parser.add_argument("--srt", type=str, required=True)
    parser.add_argument(
        "--tts", type=str, required=True, choices=["xtts_v2", "tacotron2"]
    )
    parser.add_argument(
        "--use_lipsync", type=ast.literal_eval, required=False, default=False
    )
    parser.add_argument(
        "--enhance_audio", type=ast.literal_eval, required=False, default=True
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    args = parser.parse_args()

    final_video_path, lipsync_vid_path, translated_audio_path, abs_duration_diff = (
        translate_video(
            video_file=args.i,
            srt_path=args.srt,
            use_lipsync=args.use_lipsync,
            enhance_audio=args.enhance_audio,
            tts_model=args.tts,
            target_language="de",
        )
    )

    print("\n" * 3)
    print("-" * 100)
    if lipsync_vid_path:
        print(
            f"Translated Video = {final_video_path} \n Lipsync Video = {lipsync_vid_path} \n abs_duration_diff = {abs_duration_diff}"
        )
    else:
        print(
            f"Translated Video = {final_video_path} \n abs_duration_diff = {abs_duration_diff}"
        )
    print("-" * 100)
