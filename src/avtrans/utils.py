import pysrt
import moviepy
import parselmouth
from parselmouth.praat import call
import tempfile


def load_srt(srt_path):
    subs = pysrt.open(srt_path)
    transcription = " ".join([sub.text for sub in subs])

    return transcription


def strip_audio_from_vid(video_path, audio_save_path, overwrite=False):
    video = moviepy.VideoFileClip(video_path)
    audio = video.audio

    if audio_save_path.is_file() and not overwrite:
        print("File already exists, pass overwrite = True to generate again")
        return

    audio.write_audiofile(audio_save_path)


def write_to_disk(text, path):
    with open(path, "w") as f:
        f.writelines(text)


def get_sentence_groups(srt_path):
    subs = pysrt.open(srt_path)
    groups = []
    current_group = {"text": "", "start": None, "end": None}

    for sub in subs:
        if current_group["start"] is None:
            current_group["start"] = sub.start.ordinal / 1000.0

        text = sub.text.replace("\n", " ").strip()
        current_group["text"] += text + " "
        current_group["end"] = sub.end.ordinal / 1000.0

        if text.endswith((".", "?", "!")):
            current_group["text"] = current_group["text"].strip()
            groups.append(current_group)
            current_group = {"text": "", "start": None, "end": None}

    if current_group["text"]:
        current_group["text"] = current_group["text"].strip()
        groups.append(current_group)

    return groups


def stretch_speech(input_file, output_file, factor):
    sound = parselmouth.Sound(str(input_file))
    manip = call(sound, "To Manipulation", 0.01, 75, 600)

    duration_tier = call(manip, "Extract duration tier")

    call(duration_tier, "Remove points between", 0, sound.xmax)
    call(duration_tier, "Add point", 0, factor)
    call([manip, duration_tier], "Replace duration tier")

    new_sound = call(manip, "Get resynthesis (overlap-add)")
    new_sound.save(str(output_file), "WAV")

    print(f"Saved stretched audio to: {output_file}")


def stitch_audio(audio_paths, output_path):
    audio_clips = [moviepy.AudioFileClip(str(path)) for path in audio_paths]
    final_clip = moviepy.concatenate_audioclips(audio_clips)
    final_clip.write_audiofile(output_path)


def combine_video_audio(video_path, audio_path, output_path):
    video = moviepy.VideoFileClip(video_path)
    audio = moviepy.AudioFileClip(audio_path)

    factor = video.duration / audio.duration

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio_file:
        stretched_audio_path = temp_audio_file.name
        stretch_speech(audio_path, stretched_audio_path, factor)
        audio = moviepy.AudioFileClip(stretched_audio_path)

    video = video.with_audio(audio)
    video.write_videofile(output_path, codec="libx264", audio_codec="aac")
