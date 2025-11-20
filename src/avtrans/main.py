import utils
import voice_clone
import translate
import pathlib
import torch
import tqdm
import librosa


if __name__ == "__main__":
    ## all paths
    video_path = pathlib.Path("../../data/Tanzania-2.mp4")
    srt_path = pathlib.Path("../../data/Tanzania-caption.srt")
    extracted_audio_path = video_path.with_name(
        video_path.stem + "_extracted_audio.wav"
    )
    translated_audio_path = video_path.with_name(video_path.stem + "_translated.wav")
    translated_text_path = video_path.parent / "translated_text.txt"
    final_video_path = video_path.with_name(video_path.stem + "_translated.mp4")
    chunk_paths = pathlib.Path("../../data/chunks")

    ## define our cloning and translation functions, add more models later
    voice_clone_func = voice_clone.xtts_generate_from_transcript
    translate_func = translate.helsinki_translate

    ## create dirs
    chunk_paths.mkdir(exist_ok=True)

    ## extract audio
    utils.strip_audio_from_vid(video_path, extracted_audio_path)

    ## get sentence groups, contains start, end, text
    sentence_groups = utils.get_sentence_groups(srt_path)
    untranslated_batch = [group["text"] for group in sentence_groups]

    ## translate and add to sentence groups
    translated = translate_func(untranslated_batch)
    for i in range(len(sentence_groups)):
        sentence_groups[i]["translated_text"] = translated[i]

    torch.cuda.empty_cache()

    ## generate audio for each sentence group
    audio_paths = []

    for i, group in enumerate(tqdm.tqdm(sentence_groups)):
        chunk_audio_path = chunk_paths / f"{i}.wav"

        if not chunk_audio_path.exists():
            voice_clone_func(
                group["translated_text"], extracted_audio_path, chunk_audio_path
            )

        audio_paths.append(chunk_audio_path)

    ## stretch audio to match sentence duration
    stretched_audio_paths = []

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

    ## stitch all audio and add to video
    utils.stitch_audio(stretched_audio_paths, translated_audio_path)
    utils.combine_video_audio(video_path, translated_audio_path, final_video_path)

    print(
        f"Total Duration Difference when generating audio: {round(abs_duration_diff, 2)} seconds"
    )
