"""
Downloader
"""

import os
import json
import cv2
import argparse
from pytube import Playlist, YouTube
from pytube.exceptions import VideoUnavailable
import os
import subprocess
def downloadYouTube(yt, videourl, path):
    video_stream = yt.streams.filter(progressive=False, file_extension='mp4').order_by('resolution').desc().first()
    audio_stream = yt.streams.filter(only_audio=True).order_by('abr').desc().first()
    if video_stream.fps >= 25:
        video_id = videourl.split('=')[-1]
        video_path = os.path.join(path, f"{video_id}_video.mp4")
        audio_path = os.path.join(path, f"{video_id}_audio.mp4")
        final_path = os.path.join(path, f"{video_id}.mp4")

        print("Downloading video...")
        video_stream.download(filename=video_path)
        print("Downloading audio...")
        audio_stream.download(filename=audio_path)

        print("Merging video and audio...")
        subprocess.run([
        'ffmpeg', '-i', video_path, '-i', audio_path, '-r', '25',
        '-c:v', 'copy', '-c:a', 'aac', '-strict', 'experimental',
        final_path, '-y'
        ])

        os.remove(video_path)
        os.remove(audio_path)
        return True
    else:
        return False

def process_ffmpeg(raw_vid_path, save_folder, save_vid_name,
                   bbox, time):
    """
    raw_vid_path:
    save_folder:
    save_vid_name:
    bbox: format: top, bottom, left, right. the values are normalized to 0~1
    time: begin_sec, end_sec
    """

    def secs_to_timestr(secs):
        hrs = secs // (60 * 60)
        min = (secs - hrs * 3600) // 60
        sec = secs % 60
        end = (secs - int(secs)) * 100
        return "{:02d}:{:02d}:{:02d}.{:02d}".format(int(hrs), int(min),
                                                    int(sec), int(end))

    def expand(bbox, ratio):
        top, bottom = max(bbox[0] - ratio, 0), min(bbox[1] + ratio, 1)
        left, right = max(bbox[2] - ratio, 0), min(bbox[3] + ratio, 1)

        return top, bottom, left, right

    def to_square(bbox):
        top, bottom, left, right = bbox
        h = bottom - top
        w = right - left
        c = min(h, w) // 2
        c_h = (top + bottom) / 2
        c_w = (left + right) / 2

        top, bottom = c_h - c, c_h + c
        left, right = c_w - c, c_w + c
        return top, bottom, left, right

    def denorm(bbox, height, width):
        top, bottom, left, right = \
            round(bbox[0] * height), \
            round(bbox[1] * height), \
            round(bbox[2] * width), \
            round(bbox[3] * width)

        return top, bottom, left, right

    out_path = os.path.join(save_folder, save_vid_name)

    cap = cv2.VideoCapture(raw_vid_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    top, bottom, left, right = to_square(
        denorm(expand(bbox, 0.02), height, width))
    start_sec, end_sec = time
    cmd = f"ffmpeg -i {raw_vid_path} -r 25 -vf crop=w={right-left}:h={bottom-top}:x={left}:y={top},scale=512:512 -ss {start_sec} -to {end_sec} -loglevel error {out_path}"
    os.system(cmd)


def load_data(file_path):
    with open(file_path) as f:
        data_dict = json.load(f)

    for key, val in data_dict.items():
        save_name = key+".mp4"
        ytb_id = val['youtube_id']
        time = val['duration']['start_sec'], val['duration']['end_sec']

        bbox = [val['bbox']['top'], val['bbox']['bottom'],
                val['bbox']['left'], val['bbox']['right']]
        language = val['language']
        yield ytb_id, save_name, time, bbox, language


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--language', type=str, default="dutch", help='Language')
    args = parser.parse_args()

    processed_vid_root = './processed_video'
    json_path = os.path.join('./annotations', f'{args.language}.json')  # json file path
    raw_vid_root = './raw_video'  # download raw video path

    os.makedirs(processed_vid_root, exist_ok=True)
    os.makedirs(raw_vid_root, exist_ok=True)

    video_count = 0
    for vid_id, save_vid_name, time, bbox, language in load_data(json_path):
        processed_vid_dir = os.path.join(processed_vid_root, language)
        raw_vid_dir = os.path.join(raw_vid_root, language)
        raw_vid_path = os.path.join(raw_vid_dir, vid_id + ".mp4")

        os.makedirs(processed_vid_dir, exist_ok=True)
        os.makedirs(raw_vid_dir, exist_ok=True)

        url = 'https://www.youtube.com/watch?v='+vid_id
        if not os.path.isfile(raw_vid_path) :

            yt = YouTube(url, use_oauth=True)
            success = downloadYouTube(yt, url, raw_vid_dir)

            if not success :
                continue

        process_ffmpeg(raw_vid_path, processed_vid_dir, save_vid_name, bbox, time)
        video_count = video_count + 1

    print(f"Total video_count = {video_count}")
    #os.rmdir(raw_vid_root)