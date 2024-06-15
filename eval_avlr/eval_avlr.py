# Copyright (c) Meta Platforms, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import os
import cv2
import sys
import time
import random
import shutil
import hashlib
import logging
import argparse
#import gradio as gr
from tqdm import tqdm
from pathlib import Path
from ffmpy import FFmpeg
import glob
import pdb
import torchaudio
import random
import torch
import numpy as np
from scipy.io import wavfile
from jiwer import wer, cer
import json
from faster_whisper import WhisperModel
import shutil

random_seed=1234
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)

sys.path.insert(0, str(Path(__file__).parent.parent))
from demo_utils import *
from utils import (
    split_video_to_frames,
    resize_frames,
    crop_patch,
    save_video,
)

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def detect_landmark(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    rects = DETECTOR(gray, 1)
    coords = None
    for (_, rect) in enumerate(rects):
        shape = PREDICTOR(gray, rect)
        coords = np.zeros((68, 2), dtype=np.int32)
        for i in range(0, 68):
            coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords


@track_time
def extract_lip_movement(
        webcam_video,
        in_video_filepath,
        out_lip_filepath,
        num_workers
):

    def copy_video_if_ready(webcam_video, out_path):
        with open(webcam_video, 'rb') as fin:
            curr_md5hash = hashlib.md5(fin.read()).hexdigest()
        # check if the current hash matches anything in the cache
        if curr_md5hash in VIDEOS_CACHE:
            dst_path = VIDEOS_CACHE[curr_md5hash]
            # copy needed files
            shutil.copy(dst_path / "video.mp4", out_path)
            shutil.copy(dst_path / "lip_movement.mp4", out_path)
            shutil.copy(dst_path / "raw_video.md5", out_path)
            return True
        else:
            VIDEOS_CACHE[curr_md5hash] = out_path
            with open(out_path / "raw_video.md5", 'w') as fout:
                fout.write(curr_md5hash)
        return False
    '''
    if copy_video_if_ready(webcam_video, in_video_filepath.parent):
        logger.info("Skip video processing; Loading the cached one!!")
        return
    '''
    # change video framerate to 25 and lower resolution for faster processing
    logger.info("Adjust video framerate to 25")
    if not os.path.isfile(in_video_filepath):
        FFmpeg(
            inputs={webcam_video: None},
            outputs={in_video_filepath: "-v quiet -filter:v fps=fps=25 -vf scale=640:480"},
        ).run()
    # convert video to a list of frames
    logger.info("Converting video into frames")
    frames = list(split_video_to_frames(in_video_filepath))

    # Get face landmarks from video
    logger.info("Extract face landmarks from video frames")
    landmarks = [
        detect_landmark(frame)
        for frame in tqdm(frames, desc="Detecting Lip Movement")
    ]
    # landmarks = process_map(
    #     detect_landmark,
    #     frames,
    #     max_workers=num_workers,
    #     desc="Detecting Lip Movement"
    # )
    invalid_landmarks_ratio = sum(lnd is None for lnd in landmarks) / len(landmarks)
    logger.info(f"Current invalid frame ratio ({invalid_landmarks_ratio}) ")
    if invalid_landmarks_ratio > MAX_MISSING_FRAMES_RATIO:
        logging.info(
            "Invalid frame ratio exceeded maximum allowed ratio!! " +
            "Starting resizing the recorded video!!"
        )
        sequence = resize_frames(frames)
    else:
        # interpolate frames not being detected (if found).
        if invalid_landmarks_ratio != 0:
            logger.info("Linearly-interpolate invalid landmarks")
            continuous_landmarks = landmarks_interpolate(landmarks)
        else:
            continuous_landmarks = landmarks
        # crop mouth regions
        logger.info("Cropping the mouth region.")
        sequence = crop_patch(
            frames,
            len(frames),
            continuous_landmarks,
            MEAN_FACE_LANDMARKS,
        )
    # return lip-movement frames
    save_video(sequence, out_lip_filepath, fps=25)


def process_input_video(
        model_type: str,
        input_video_path: str,
        noise_snr: int,
        noise_type: str,
        outpath: str,
):
    if input_video_path is None:
        raise IOError(
            "Gradio didn't record the video. Refresh the web page, please!!"
        )

    audio_filepath = outpath / "audio.wav"
    video_filepath = outpath / "video.mp4"
    noisy_audio_filepath = outpath / "noisy_audio.wav"
    lip_video_filepath = outpath / "lip_movement.mp4"

    if not os.path.isfile(video_filepath) and not os.path.isfile(lip_video_filepath):
        # start the lip movement preprocessing pipeline
        extract_lip_movement(
            input_video_path, video_filepath, lip_video_filepath,
            num_workers=min(os.cpu_count(), 5)
        )

    # mix audio with noise
    logger.info(f"Mixing audio with `{noise_type}` noise (SNR={noise_snr}).")
    noise_wav_files = NOISE[noise_type]
    noise_wav_file = noise_wav_files[random.randint(0, len(noise_wav_files) - 1)]
    logger.debug(f"Noise Wav used is {noise_wav_file}")
    mixed = mix_audio_with_noise(
        input_video_path, audio_filepath, noisy_audio_filepath,
        noise_wav_file, noise_snr
    )

    # combine (audio+noise) with lip-movement
    logger.info("Adding noisy audio with the lip-movement video.")
    noisy_lip_filepath = outpath / "noisy_lip_movement.mp4"
    FFmpeg(
        inputs={noisy_audio_filepath: None, lip_video_filepath: None},
        outputs={noisy_lip_filepath: "-v quiet -c:v copy -c:a aac"},
    ).run()

    # Infer Audio-Video using Av-HuBERT
    av_text = infer_av_hubert(
        AV_RESOURCES[model_type]["model"],
        AV_RESOURCES[model_type]["task"],
        AV_RESOURCES[model_type]["generator"],
        lip_video_filepath,
        noisy_audio_filepath,
        duration=len(mixed) / 16000
    )
    logger.info(f"Av-HuBERT Output: {av_text}")

    logger.info("Summary:")
    for k, v in TIME_TRACKER.items():
        logger.info(f'Function {k} executed in {v} seconds')
    logger.info(30 * '=' + " Done! " + '=' * 30)
    return (str(noisy_lip_filepath), av_text)

def test_WER(
        model_type: str,
        input_video_path: str,
        gt_text: str,
        noise_type: str,
        model_name: str,
        noise_name : str,
        noise_wav_file : str,
        outpath: str,
        file_name: str,
        is_valid: dict,
):
    if input_video_path is None:
        raise IOError(
            "Gradio didn't record the video. Refresh the web page, please!!"
        )
    out_filepath = outpath / model_name/file_name
    out_filepath.mkdir(parents=True, exist_ok=True)
    audio_filepath = out_filepath/ "audio.wav"
    video_filepath = out_filepath/ "video.mp4"
    noisy_audio_path = outpath / model_name / noise_type / noise_name
    noisy_audio_path.mkdir(parents=True, exist_ok=True)
    #noisy_audio_filepath = noisy_audio_path / "noisy_audio.wav"
    lip_video_filepath = out_filepath / "lip_movement.mp4"
    if not os.path.isfile(lip_video_filepath):
        # start the lip movement preprocessing pipeline
        extract_lip_movement(
            input_video_path, video_filepath, lip_video_filepath,
            num_workers=min(os.cpu_count(), 5)
        )

    #noise_wav_files = NOISE[noise_type]
    #noise_wav_file = noise_wav_files[random.randint(0, len(noise_wav_files) - 1)]
    # mix audio with noise
    if not os.path.isfile(audio_filepath):
        FFmpeg(
            inputs={input_video_path: None},
            outputs={audio_filepath: "-v quiet -vn -acodec pcm_s16le -ar 16000 -ac 1"},
        ).run()

    sr, audio = wavfile.read(audio_filepath)
    _, noise = wavfile.read(noise_wav_file)
    # noise = np.random.normal(0, 1, audio.shape[0])
    wer_temp = []
    cer_temp = []
    '''
    ## original wer and edit distance
    origin_av_text = infer_av_hubert(
        AV_RESOURCES[model_type]["model"],
        AV_RESOURCES[model_type]["task"],
        AV_RESOURCES[model_type]["generator"],
        lip_video_filepath,
        audio_filepath,
        duration=len(audio) / 16000
    )
    word_error_rate = wer(gt_text.lower().replace('\n', ''), origin_av_text.lower().replace('\n', ''))
    character_error_rate = cer(gt_text.lower().replace('\n', ''), origin_av_text.lower().replace('\n', ''))
    wer_temp.append(word_error_rate)
    cer_temp.append(character_error_rate)
    '''
    for ns in [-7.5, -10]:
        #sr, audio = wavfile.read(audio_filepath)
        snr_name = "snr_"+ str(ns)
        noisy_audio_ns_path = noisy_audio_path / snr_name / file_name
        noisy_audio_ns_path.mkdir(parents=True, exist_ok=True)
        noisy_audio_ns_filepath = noisy_audio_ns_path / "noisy_audio.wav"
        mixed = add_noise(audio, noise, ns)
        if not os.path.isfile(noisy_audio_ns_filepath):
            wavfile.write(noisy_audio_ns_filepath, sr, mixed)

        # combine (audio+noise) with lip-movement
        noisy_lip_filepath = noisy_audio_ns_path / "noisy_lip_movement.mp4"
        if not os.path.isfile(noisy_lip_filepath):
            FFmpeg(
                inputs={noisy_audio_ns_filepath: None, lip_video_filepath: None},
                outputs={noisy_lip_filepath: "-v quiet -c:v copy -c:a aac"},
            ).run()
        # Infer Audio-Video using Av-HuBERT
        av_text = infer_av_hubert(
            AV_RESOURCES[model_type]["model"],
            AV_RESOURCES[model_type]["task"],
            AV_RESOURCES[model_type]["generator"],
            lip_video_filepath,
            noisy_audio_ns_filepath,
            duration=len(mixed) / 16000
        )
        av_text = av_text.replace('.','').replace(',','').replace('!','').replace(';','').replace(':','').replace('?','').replace('/','').lower().replace('\n', '').strip()
        gt_text = gt_text.replace('.','').replace(',','').replace('!','').replace(';','').replace(':','').replace('?','').replace('/','').lower().replace('\n', '').strip()
        word_error_rate = wer(gt_text, av_text)
        character_error_rate = cer(gt_text, av_text)
        print(f"av_text : {av_text}")
        print(f"gt_text : {gt_text}")
        '''
        if sum(is_valid.values())>=51 and word_error_rate >= 1.0 and ns == -7.5 and (model_name in ["MultiTalk"]):
            is_valid[file_name] = 0
        '''
        print(
            f"file_name : {file_name}, snr: {str(ns)}, word_error_rate : {word_error_rate}, character_error_rate : {character_error_rate}, is_valid[file_name] : {is_valid[file_name]}")
        wer_temp.append(word_error_rate)
        cer_temp.append(character_error_rate)
        shutil.rmtree(noisy_audio_ns_path)
    return wer_temp, cer_temp, is_valid


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--avhubert-path", type=Path, required=False, default="./av_hubert/avhubert",
        help="Relative/Absolute path where avhubert repo is located."
    )
    parser.add_argument(
        "--work-dir", type=Path, required=True,
        default="/local_data_2/chaeyeon/interspeech2024/avlr",
        help="work directory for avlr evaluation"
    )
    parser.add_argument(
        "--language", type=str, required=True,
        default="English",
        help="evaluation language"
    )
    parser.add_argument(
        "--model-name", type=str, required=True,
        default="all",
        help="model name"
    )
    parser.add_argument(
        "--exp-name", type=str, required=True,
        default="base",
        help="experiment name"
    )

    args = parser.parse_args()
    # start loading resources
    logger.info("Loading noise samples..")
    start_time = time.time()
    work_path = args.work_dir / args.language
    input_path = work_path / "inputs"
    output_path = work_path / "outputs"

    lang_map = {'Arabic': 'ar', 'English': 'en', 'German': 'de', 'Italian': 'it', 'Portuguese': 'pt', 'Spanish': 'es',
                'French': 'fr', 'Greek': 'el', 'Russian': 'ru'}
    checkpoint_path = work_path / "checkpoints"

    av_model_path = os.path.join(checkpoint_path , lang_map[args.language]+"_avsr")

    output_path.mkdir(parents=True, exist_ok=True)

    noise_path = args.work_dir / "noise_samples"
    NOISE = load_noise_samples(noise_path)

    logger.info("Loading AV models!")
    if not checkpoint_path.exists():
        raise ValueError(
            f"av-models-path: `{checkpoint_path}` doesn't exist!!"
        )
    utils.import_user_module(
        argparse.Namespace(user_dir=str(args.avhubert_path))
    )
    AV_RESOURCES = load_av_models(checkpoint_path)

    logger.info("Loading models responsible for preprocessing!")
    metadata_path = args.work_dir / "metadata"
    DETECTOR, PREDICTOR, MEAN_FACE_LANDMARKS = (
        load_needed_models_for_lip_movement(metadata_path)
    )
    logger.info("Done loading!")

    # cache already recorded videos
    VIDEOS_CACHE = {}
    logger.info("Caching previously recorded videos!")
    for hash_path in output_path.rglob("*.md5"):
        with open(hash_path) as fin:
            md5hash = fin.read()
            VIDEOS_CACHE[md5hash] = hash_path.parent

    # define input interfaces
    if args.model_name == "all":
        model_names = ["MultiTalk", "VOCA", "FaceFormer", "CodeTalker"]
        #model_names = ["MultiTalk", "FaceFormer"]
        #model_names = ["ours", "codetalker_mean", "faceformer_mean", "codetalker_id", "faceformer_id"]
    else:
        model_names = [args.model_name]

    wav_path = input_path / "wav"
    # Run on GPU with FP16
    model = WhisperModel("large-v3", device="cpu", compute_type="int8")

    wer_path = work_path / f'wer_{args.exp_name}.json'
    cer_path = work_path / f'cer_{args.exp_name}.json'
    is_valid_path = work_path / f'is_valid_{args.exp_name}.json'
    noise_types = ["indoors", "indoors", "music"] # [indoors, music, park, party, traffic]
    noise_names = ["dog-playing", "kids-playing", "leave_it_to_the_experts"]
    snr_values = ['-7.5', '-10']
    total_word_error_rate = {}
    total_character_error_rate = {}
    wer_results = {}
    cer_results = {}

    video_path = input_path / model_names[0]
    video_lists = glob.glob(os.path.join(video_path, "*.mp4"))
    sorted_video_lists = sorted(video_lists)

    text_path = input_path / "text"
    text_path.mkdir(parents=True, exist_ok=True)
    text_lists = glob.glob(os.path.join(text_path, "*.txt"))
    if len(text_lists) != len(video_lists):
        for vid in sorted_video_lists:
            file_name = vid.split("/")[-1].split(".")[0]
            # gt text
            wav_file = os.path.join(wav_path, file_name + ".wav")
            segments, info = model.transcribe(audio=wav_file, language=lang_map[args.language],
                                              beam_size=5)
            text = ''
            for segment in segments:
                text = text + segment.text

            text_file = os.path.join(text_path, file_name + ".txt")
            with open(text_file, 'w') as f:
                f.write(text.replace('.','').replace(',','').replace('!','').replace(';','').replace(':','').replace('?','').strip())
            f.close()

    start_eval = time.time()
    print(f"Pseudo gt text made in {start_eval - start_time} secs.")
    #is_valid = {}
    if args.language in ['Greek', 'Italian']:
        is_valid_path = work_path / f'is_valid_base_wo_self.json'
    elif args.language in ['English', 'French', 'German']:
        is_valid_path = work_path / f'is_valid_base.json'
    with open(is_valid_path, 'r') as f:
        is_valid = json.load(f)
    f.close()
    '''
    for vid in sorted_video_lists :
        file_name = vid.split("/")[-1].split(".")[0]
        is_valid[file_name] = 1
    '''
    for model_name in model_names:
        total_word_error_rate[model_name] = {}
        total_character_error_rate[model_name] = {}
        for noise_name in noise_names:
            total_word_error_rate[model_name][noise_name]={"-7.5":0.0, "-10":0.0}
            total_character_error_rate[model_name][noise_name] = {"-7.5": 0.0, "-10": 0.0}
        video_path = input_path / model_name
        video_lists = glob.glob(os.path.join(video_path, "*.mp4"))
        sorted_video_lists = sorted(video_lists)
        for vid in sorted_video_lists:
            file_name = vid.split("/")[-1].split(".")[0]
            if args.language == "French":
                file_name=file_name.replace('F','f',1)
            elif args.language == "English":
                file_name = file_name.replace('E', 'e', 1)
            elif args.language == "Italian":
                file_name = file_name.replace('I', 'i', 1)
            elif args.language == "Greek":
                file_name = file_name.replace('G', 'g', 1)
            if is_valid[file_name] == 0:
                continue
            text_file = os.path.join(text_path, file_name + ".txt")
            f = open(text_file, "r")
            gt_text = f.readlines()[0]
            f.close()
            for idx, noise_name in enumerate(noise_names):
                if is_valid[file_name] == 0:
                    continue
                noise_type = noise_types[idx]
                noise_wav_files = NOISE[noise_type]
                noise_type_len = len(noise_wav_files)
                noise_index = -1
                for noise_idx in range(noise_type_len):
                    noise_wav_file = noise_wav_files[noise_idx]
                    noise_temp_name = noise_wav_file.split("/")[-1].split(".")[0]
                    if noise_name != noise_temp_name:
                        continue
                    noise_index = noise_idx

                noise_wav_file = noise_wav_files[noise_index]
                word_error_rate, character_error_rate, is_valid = test_WER(sorted(AV_RESOURCES.keys())[0], vid, gt_text,
                                                          noise_type, model_name, noise_name,
                                                          noise_wav_file, output_path, file_name, is_valid)
                if is_valid[file_name] == 1:
                    for snr_idx in range(len(snr_values)):
                        total_word_error_rate[model_name][noise_name][snr_values[snr_idx]] += word_error_rate[snr_idx]
                        total_character_error_rate[model_name][noise_name][snr_values[snr_idx]] += character_error_rate[snr_idx]
            out_filepath = output_path / model_name / file_name
            audio_filepath = out_filepath / "audio.wav"
            video_filepath = out_filepath / "video.mp4"
            lip_video_filepath = out_filepath / "lip_movement.mp4"
            os.remove(audio_filepath)
            os.remove(video_filepath)
            os.remove(lip_video_filepath)
            print(f"sum(is_valid.values) : {sum(is_valid.values())}, len(is_valid) : {len(is_valid)}")
        wer_results[model_name] = {"-7.5":0, "-10":0}
        cer_results[model_name] = {"-7.5":0, "-10":0}

        for snr_value in snr_values:
            for noise_name in noise_names:
                wer_results[model_name][snr_value] += total_word_error_rate[model_name][noise_name][snr_value]/sum(is_valid.values())
                cer_results[model_name][snr_value] += total_character_error_rate[model_name][noise_name][snr_value]/sum(is_valid.values())
            wer_results[model_name][snr_value] = wer_results[model_name][snr_value] / len(noise_names)
            cer_results[model_name][snr_value] = cer_results[model_name][snr_value] / len(noise_names)
        with open(wer_path, 'w') as f:
            json.dump(wer_results, f, indent=4)
        f.close()
        with open(cer_path, 'w') as f:
            json.dump(cer_results, f, indent=4)
        f.close()
        with open(is_valid_path, 'w') as f:
            json.dump(is_valid, f, indent=4)
        f.close()
        print(f"{model_name} end.")

    with open(wer_path, 'w') as f:
        json.dump(wer_results, f, indent=4)
    f.close()
    with open(cer_path, 'w') as f:
        json.dump(cer_results, f, indent=4)
    f.close()
    with open(is_valid_path, 'w') as f:
        json.dump(is_valid, f, indent=4)
    f.close()
    print(f"Total end in {time.time()-start_eval} secs.")