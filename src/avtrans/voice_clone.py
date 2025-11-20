import torch
import torchaudio as ta

default_device = "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def chatterbox_generate_from_transcript(
    transcript,
    styling_prompt_path,
    output_path,
    language_id=None,
    device=default_device,
):
    from chatterbox.tts import ChatterboxTTS
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS

    kwargs = {}
    if language_id:
        model = ChatterboxMultilingualTTS.from_pretrained(device=device)
        kwargs = {"language_id": language_id}
    else:
        model = ChatterboxTTS.from_pretrained(device=device)

    wav = model.generate(transcript, audio_prompt_path=styling_prompt_path, **kwargs)
    ta.save(output_path, wav, model.sr)

    del model
    torch.cuda.empty_cache()


@torch.no_grad()
def xtts_generate_from_transcript(
    transcript, styling_prompt_path, output_path, device=default_device
):
    from TTS.api import TTS

    tts = TTS("tts_models/de/thorsten/tacotron2-DDC").to(device)
    # tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=True).to(device)

    tts.tts_with_vc_to_file(
        transcript, speaker_wav=styling_prompt_path, file_path=output_path
    )
