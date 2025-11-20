import utils
import voice_clone
import translate
import pathlib
import torch
import tqdm
import librosa


if __name__ == "__main__":
    video_path = pathlib.Path("../../data/Tanzania-2.mp4")
    srt_path = pathlib.Path("../../data/Tanzania-caption.srt")
    extracted_audio_path = video_path.with_name(
        video_path.stem + "_extracted_audio.wav"
    )
    translated_audio_path = video_path.with_name(video_path.stem + "_translated.wav")
    translated_text_path = video_path.parent / "translated_text.txt"
    final_video_path = video_path.with_name(video_path.stem + "_translated.mp4")

    utils.strip_audio_from_vid(video_path, extracted_audio_path)

    sentence_groups = utils.get_sentence_groups(srt_path)

    untranslated_batch = [group["text"] for group in sentence_groups]
    translated = translate.helsinki_translate(untranslated_batch)
    for i in range(len(sentence_groups)):
        sentence_groups[i]["translated_text"] = translated[i]

    torch.cuda.empty_cache()

    chunk_paths = pathlib.Path("../../data/chunks")
    chunk_paths.mkdir(exist_ok=True)

    audio_paths = []

    for i, group in enumerate(tqdm.tqdm(sentence_groups)):
        chunk_audio_path = chunk_paths / f"{i}.wav"

        if not chunk_audio_path.exists():
            voice_clone.xtts_generate_from_transcript(
                group["translated_text"], extracted_audio_path, chunk_audio_path
            )

        audio_paths.append(chunk_audio_path)

    combined_audio_data = []
    stretched_audio_paths = []

    sample_rate = None

    abs_duration_diff = 0
    for i, audio_path in tqdm.tqdm(enumerate(audio_paths)):
        actual_duration = sentence_groups[i]["end"] - sentence_groups[i]["start"]
        current_duration = librosa.get_duration(path=audio_path)
        utils.stretch_speech(
            audio_path,
            chunk_paths / f"{i}_stretched.wav",
            actual_duration / current_duration,
        )
        stretched_audio_paths.append(chunk_paths / f"{i}_stretched.wav")
        abs_duration_diff += abs(actual_duration - current_duration)

    utils.stitch_audio(stretched_audio_paths, translated_audio_path)
    utils.combine_video_audio(video_path, translated_audio_path, final_video_path)

    print(
        f"Total Duration Difference when generating audio: {round(abs_duration_diff, 2)} seconds"
    )
