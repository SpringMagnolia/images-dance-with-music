import logging
import os
import random

import cv2
import moviepy.editor as mpy
import numpy as np
from scipy.io import wavfile
from tqdm import tqdm

from utils import merge_audio_video, to_wav

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.WARNING)


def audio_normalize(audio_path, std_thresh=3):
    # 变成wav格式
    new_path = to_wav(audio_path)
    rate, data = wavfile.read(new_path)
    one_channel_data = np.mean(data, axis=1)
    # 3倍标准差
    thresh = std_thresh*one_channel_data.std()

    def down_thresh(i):
        if i > 0:
            return i-thresh if i-thresh > 0 else 0
        elif i < 0:
            return i+thresh if i+thresh < 0 else 0
        else:
            return 0
    v_func = np.vectorize(down_thresh)

    normalized_data = v_func(one_channel_data)
    return normalized_data


def get_audio_clip_time(normalized_audio, min_size=200, rate=44100):
    min_size = int(min_size/1000 * rate)
    length = normalized_audio.shape[0]
    t = 0
    exist = False
    continue_zeros = set()

    while t <= length-min_size:
        if np.abs(normalized_audio[t:t+min_size]).sum() == 0:
            exist = False
        elif exist == False:
            exist = True
            continue_zeros.add(t-1)
        else:  # 中间存在!=0
            pass
        t += 100  # 这里的可以减少到1，更准确但是耗时更久

    ret = list(continue_zeros)
    ret.sort()
    ret = [int(i/rate * 1000) for i in ret]
    return ret, length/rate * 1000


def resize_crop_image(image_path):
    image = cv2.imread(image_path)
    h, w, _ = image.shape
    # 进行resize
    ratio = min(1080 / h, 1920 / w)
    changed_h = True if (1080 / h) == ratio else False
    if changed_h:
        new_h = 1080
        new_w = int(ratio * w)
    else:
        new_w = 1920
        new_h = int(ratio * h)

    resized = cv2.resize(image, (new_w, new_h))
    # 进行padding，其实这里可以进行crop操作，实现图片没有黑边
    top = (1080 - new_h) // 2
    bottom = (1080 - new_h) // 2
    if top + bottom + new_h < 1080:
        bottom += 1
    elif top + bottom + new_h > 1080:
        if top > 0:
            top -= 1
        else:
            bottom -= 1

    left = (1920 - new_w) // 2
    right = (1920 - new_w) // 2
    if left + right + new_w < 1920:
        right += 1
    elif left + right + new_w > 1920:
        if left > 0:
            left -= 1
        else:
            right -= 1

    pad_image = cv2.copyMakeBorder(
        resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    return pad_image

def video_maker(audio_clip_list, audio_path, audio_length, img_path):
    image_list = [os.path.join(img_path, i) for i in os.listdir(img_path)]

    audioclip = mpy.AudioFileClip(audio_path)
    audio_clip_list.insert(0, 0)
    audio_subclips = []
    audio_durations = []
    for i in range(len(audio_clip_list)-1):
        duration = (audio_clip_list[i+1] - audio_clip_list[i])/1000
        audio_durations.append(duration)
    audio_durations.append((audio_clip_list[-1]-audio_clip_list[-2])/1000)
    audio_durations.append((audio_length-audio_clip_list[-1])/1000)

    choosed_images = random.sample(image_list, len(audio_durations))

    choosed_images_array = []
    for i in tqdm(choosed_images, desc="正在处理图片"):
        choosed_images_array.append(resize_crop_image(i))

    clip = mpy.ImageSequenceClip(
        choosed_images_array, durations=audio_durations)
    clip = clip.subclip(0, round(audio_length/1000, 2))

    # clip.set_audio(audioclip)

    file_path, file_name = audio_path.rsplit("/", 1)
    output_path = file_path + "/"+file_name.split(".")[0]+"/output.mp4"
    logger.warning("正在将图片转为视频...")

    clip.write_videofile(output_path, fps=30, codec='libx264',
                         audio_codec='aac', temp_audiofile='temp-audio.m4a', remove_temp=True)
    logger.warning("图片转视频结束：{}".format(output_path))
    return output_path


if __name__ == "__main__":
    audio_path = "./music/bad.mp3"
    img_path = "./spider/alpha/images/full"
    std_thresh = 2
    step_min_size = 50

    normalized_audio = audio_normalize(audio_path, std_thresh=std_thresh)
    audio_clip_list, audio_length = get_audio_clip_time(
        normalized_audio, min_size=step_min_size)
    video_output_path = video_maker(
        audio_clip_list, audio_path, audio_length, img_path)

    logger.warning("正在将音频和视频进行合成")
    output_path = merge_audio_video(video_output_path, audio_path, version=1)
    logger.warning("合成完成：最终视频地址在:{}".format(output_path))
