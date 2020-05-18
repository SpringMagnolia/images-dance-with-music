import os
from pprint import pprint

import ffmpeg
from pydub import AudioSegment, silence, utils


def get_mediainfo(file_path):
    return utils.mediainfo_json(file_path)


def merge_audio_video(video_path, audio_path,version=2):
    audio_name = os.path.basename(audio_path).split(".")[0]
    output_path = os.path.join(os.path.dirname(
        video_path), audio_name+"-video-audio-version{}.mp4".format(version))
    input_video = ffmpeg.input(video_path)
    input_audio = ffmpeg.input(audio_path)
    ffmpeg.concat(input_video, input_audio, v=1, a=1).output(output_path).global_args('-loglevel', 'error').global_args('-y').run()
    return output_path

def to_wav(file_path):
    song = AudioSegment.from_file(file_path, file_path.split(".")[-1])
    song_name = file_path.rsplit("/",1)[-1].split(".")[0]

    new_path = file_path.rsplit(".", 1)[0]+"/"+song_name+'.wav'
    dirname = os.path.dirname(new_path)
    if not os.path.exists(dirname):
        os.mkdir(dirname)
    song.export(new_path, format("wav"))
    return new_path


if __name__ == "__main__":
    ret = get_mediainfo("./test_music/Señorita.wav")
    pprint(ret)
