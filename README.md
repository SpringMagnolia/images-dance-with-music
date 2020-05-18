# README

## 环境安装
分别执行下面命令安装环境

```sh
pip install moviepy
pip install tqdm
conda install spleeter
conda install scrapy
conda install scipy
pip install pydub
pip install opencv-python
pip install ffmpeg-python
conda install tensorflow
```

## 使用方法：

#### 1. 有鼓点的音乐
运行`python video_maker-v2.py`进行体验

其中，下列参数可以直接进行修改

```python
if __name__ == "__main__":
    audio_path = "./music/rose.mp3"
    img_path = "./spider/alpha/images/full"
    std_thresh = 2 
    step_min_size = 50
```
`audio_path` 表示音乐路径
`img_path` 表示图片文件夹
`std_thresh` 使用几倍校准差进行鼓点清理
`step_min_size` 表示鼓点间的最小间隔

#### 2. 无鼓点的音乐(效果不好)
运行`python video_maker.py`进行体验
参数和上述相同

#### 3. 图片抓取
在spider/alpha下运行 `scrapy crawl fateCrawl` 即可，最终图片会保存在spider/alpha/images/full中

如果需要抓取其他图片，修改`spider/alpha/alpha/spider/fateCrawl.py`中的URL地址`https://wall.alphacoders.com/search.php?search=fate&lang=Chinese`中的search字段即可
