import logging
import os
import random

import cv2
import moviepy.editor as mpy
import numpy as np
import pydub
import tensorflow as tf
from scipy.io import wavfile
from spleeter.separator import Separator
from tqdm import tqdm

from utils import get_mediainfo, merge_audio_video

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.WARNING)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #屏蔽tensorflow的日志
tf.get_logger().setLevel("ERROR")


def split_audio(audio_path, bit_rate, duration, sample_rate):
    file_path, file_name = audio_path.rsplit("/", 1)
    new_file_path = file_path + "/"+file_name.split(".")[0]
    bit_rate = str(int(bit_rate)//1000)+"k"
    # offset=0, duration=600., codec='wav', bitrate='128k',
    logger.warning("正在分离音频，这可能需要点时间:bit_rate:{} duration:{} sample_rate:{}".format(bit_rate, duration, sample_rate))
    separator = Separator('spleeter:4stems')
    separator.separate_to_file(
        audio_path, new_file_path, duration=duration, bitrate=bit_rate)
    drum_path = "{}/{}/{}".format(new_file_path,
                                  file_name.split(".")[0], "drums.wav")
    logger.warning("分离音频结束,drum_path:{}".format(drum_path))
    return drum_path





def audio_normalize(audio_drum_path, std_thresh=3):
    # 变成wav格式
    rate, data = wavfile.read(audio_drum_path)
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


def get_audio_clip_time(normalized_audio, min_size=200, sample_rate=44100):
    min_size = int(min_size/1000 * sample_rate)
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
        t += 100  # 差不多是每次2ms递增计算

    ret = list(continue_zeros)
    ret.sort()
    ret = [int(i/sample_rate * 1000) for i in ret]
    return ret


def resize_crop_image(image_path):
    image = cv2.imread(image_path)
    h, w, _ = image.shape
    #进行resize
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

    pad_image = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    return pad_image


def video_maker_2(audio_clip_list, audio_path, audio_length, img_path):
    audio_length = audio_length*1000
    image_list = [os.path.join(img_path, i) for i in os.listdir(img_path)]

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
        try:
            padding_img = resize_crop_image(i) #偶尔图片错误，则重新选择一张图片
        except :
            padding_img = resize_crop_image(random.choice(image_list))
        
        choosed_images_array.append(padding_img)

    clip = mpy.ImageSequenceClip(
        choosed_images_array, durations=audio_durations)
    clip = clip.subclip(0, round(audio_length/1000, 2))

    # 添加音频 使用pymovie方法添加，无法成功，原因音视频长度不对
    # audioclip = mpy.AudioFileClip(audio_path)
    # clip.set_audio(audioclip)

    file_path, file_name = audio_path.rsplit("/", 1)
    output_path = file_path + "/"+file_name.split(".")[0]+"/output.mp4"
    logger.warning("正在将图片转为视频...")
    clip.write_videofile(output_path, fps=30, codec='libx264',
                         audio_codec='aac', temp_audiofile='temp-audio.m4a', remove_temp=True)
    logger.warning("图片转视频结束：{}".format(output_path))
    return output_path


if __name__ == "__main__":
    audio_path = "./music/shape.mp3"
    img_path = "./spider/alpha/images/full"
    std_thresh = 3 #使用几倍校准差进行
    step_min_size = 50

    #获取音频信息
    media_info = pydub.utils.mediainfo_json(audio_path)
    
    bit_rate = media_info["format"]["bit_rate"] #比特率
    duration = float(media_info["format"]["duration"]) #时常 s
    sample_rate = media_info["streams"][0]["sample_rate"] #采样率
    
    #分离鼓点
    audio_drum_path = split_audio(audio_path, bit_rate, duration, sample_rate)
    #对鼓点进行清理
    normalized_audio_drum = audio_normalize(audio_drum_path, std_thresh=std_thresh)
    #提取鼓点时间点
    audio_clip_list = get_audio_clip_time(
        normalized_audio_drum, min_size=step_min_size, sample_rate=int(sample_rate))
    #进行视频合成
    video_output_path = video_maker_2(audio_clip_list, audio_path, duration, img_path)
    #把音频和视频进行合成，误差有20ms左右，建议手动合成
    # video_output_path = "./music/shape/output.mp4"
    logger.warning("正在将音频和视频进行合成")
    output_path = merge_audio_video(video_output_path,audio_path)
    logger.warning("合成完成：最终视频地址在:{}".format(output_path))
