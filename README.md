##
This library performs video translation from English to German. Currently it requires a subtitle file to be provided for the input video. It uses [coqui tts](https://github.com/idiap/coqui-ai-TTS) for voice cloning and [latent sync](https://github.com/ByteDance/LatentSync) for lip sync.

## Setup
```
git clone https://github.com/Sambhav300899/VideoTranslation.git
git submodule update --init --recursive
uv pip install -e .
uv pip install -r requirements.txt
```

## Usage

Use the gradio demo to launch the demo, use the following command:
```
python gradio_demo.py
```

Alternatively, you can use the following command to translate a video:

```
python run_translation.py --i <input_video_path> --o <output_video_path> --srt <subtitle_path> --tts <tts_model> --use_lipsync <use_lipsync> --enhance_audio <enhance_audio>

# Sample command
python run_translation.py --i sample_data/Tanzania-2.mp4 --o sample_data/Tanzania-2_translated.mp4 --srt sample_data/Tanzania-caption.srt --tts xtts_v2 --use_lipsync True --enhance_audio True
```

## Sample outputs


Notes on voice cloning models

1. Chatterbox TTS - Has a very distinct english accent for German speech

2.


hf download ByteDance/LatentSync-1.5 whisper/tiny.pt --local-dir checkpoints
hf download ByteDance/LatentSync-1.5 latentsync_unet.pt --local-dir checkpoints


https://medium.com/@joshiprerak123/transform-your-audio-denoise-and-enhance-sound-quality-with-python-using-pedalboard-24da7c1df042

https://github.com/resemble-ai/resemble-enhance

https://github.com/RVC-Boss/GPT-SoVITS

https://github.com/SWivid/F5-TTS/tree/main/src/f5_tts/infer


2.8.0+cu126