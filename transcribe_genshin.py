# coding=gbk
import os
import argparse
import librosa
import numpy as np
from multiprocessing import Pool, cpu_count

import soundfile
from scipy.io import wavfile
from tqdm import tqdm
from config import config

global speaker_annos
speaker_annos = []

def process(item):  
    spkdir, wav_name, args, speaker = item
    # speaker = spkdir.replace("\\", "/").split("/")[-1]
    wav_path = os.path.join(args.in_dir, speaker, wav_name)
    if os.path.exists(wav_path) and '.wav' in wav_path:
        os.makedirs(os.path.join(args.out_dir, speaker), exist_ok=True)
        wav, sr = librosa.load(wav_path, sr=args.sr)
        soundfile.write(
            os.path.join(args.out_dir, speaker, wav_name),
            wav,
            sr
        )

def process_text(item):
    spkdir, wav_name, args, lang, speaker = item
    # speaker = spkdir.replace("\\", "/").split("/")[-1]
    wav_path = os.path.join(args.in_dir, speaker, wav_name)
    global speaker_annos
    tr_name = wav_name.replace('.wav', '')
    with open(args.in_dir+'/'+tr_name+'.lab', "r", encoding="utf-8") as file:
             text = file.read()
    text = text.replace("{NICKNAME}",'旅行者')
    text = text.replace("{M#他}{F#她}",'他')
    text = text.replace("{M#她}{F#他}",'他')
    substring = "{M#妹妹}{F#哥哥}"  
    if substring in text:
        if tr_name.endswith("a"):
           text = text.replace("{M#妹妹}{F#哥哥}",'妹妹')
        if tr_name.endswith("b"):
           text = text.replace("{M#妹妹}{F#哥哥}",'哥哥')
    text = text.replace("#",'')   
    text = f'{lang}|{text}\n' #
    speaker_annos.append(args.out_dir + '/' + wav_name + "|" + speaker + "|" + text)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sr", type=int, default=44100, help="sampling rate")
    parser.add_argument("--in_dir", type=str, default=config.resample_config.in_dir, help="path to source dir")
    parser.add_argument("--out_dir", type=str, default=config.resample_config.out_dir, help="path to target dir")
    parser.add_argument("--speaker", type=str, default="hutao", help="name of speaker")
    parser.add_argument("--language", type=str, default="ZH", help="chosen language")
    parent_dir = config.resample_config.in_dir
    speaker_names = list(os.walk(parent_dir))[0][1]   
    args = parser.parse_args()
    speaker = args.speaker
    lang = args.language
    # processs = 8
    processs = cpu_count()-2 if cpu_count() > 4 else 1
    pool = Pool(processes=processs)

    spk_dir = args.in_dir
    if os.path.isdir(spk_dir):
        print(spk_dir)
        for i in os.listdir(spk_dir):
            if i.endswith("wav"):
                pro = (spk_dir, i, args, lang, speaker)
                process_text(pro)

    if len(speaker_annos) == 0:
        print("transcribe error. len(speaker_annos) == 0")
    else:
      with open(config.preprocess_text_config.transcription_path, 'w', encoding='utf-8') as f:
        for line in speaker_annos:
            f.write(line)
      print("transcribe lab texts finished")
