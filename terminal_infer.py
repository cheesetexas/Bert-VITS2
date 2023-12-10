import argparse
from pydub import AudioSegment
import os
import logging
import re_matching
from tools.sentence import split_by_language

logging.getLogger("numba").setLevel(logging.WARNING)
logging.getLogger("markdown_it").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)

logging.basicConfig(
    level=logging.INFO, format="| %(name)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)

import torch
import utils
from infer import infer, latest_version, get_net_g, infer_multilang
import gradio as gr
import webbrowser
import numpy as np
from config import config
from tools.translate import translate
import librosa

net_g = None

device = config.webui_config.device
if device == "mps":
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


def generate_audio(
    slices,
    sdp_ratio,
    noise_scale,
    noise_scale_w,
    length_scale,
    speaker,
    language,
    reference_audio,
    emotion,
    skip_start=False,
    skip_end=False,
):
    audio_list = []
    # silence = np.zeros(hps.data.sampling_rate // 2, dtype=np.int16)
    with torch.no_grad():
        for idx, piece in enumerate(slices):
            skip_start = (idx != 0) and skip_start
            skip_end = (idx != len(slices) - 1) and skip_end
            audio = infer(
                piece,
                reference_audio=reference_audio,
                emotion=emotion,
                sdp_ratio=sdp_ratio,
                noise_scale=noise_scale,
                noise_scale_w=noise_scale_w,
                length_scale=length_scale,
                sid=speaker,
                language=language,
                hps=hps,
                net_g=net_g,
                device=device,
                skip_start=skip_start,
                skip_end=skip_end,
            )
            audio16bit = gr.processing_utils.convert_to_16_bit_wav(audio)
            audio_list.append(audio16bit)
            # audio_list.append(silence)  # 将静音添加到列表中
    return audio_list


def generate_audio_multilang(
    slices,
    sdp_ratio,
    noise_scale,
    noise_scale_w,
    length_scale,
    speaker,
    language,
    reference_audio,
    emotion,
    skip_start=False,
    skip_end=False,
):
    audio_list = []
    # silence = np.zeros(hps.data.sampling_rate // 2, dtype=np.int16)
    with torch.no_grad():
        for idx, piece in enumerate(slices):
            skip_start = (idx != 0) and skip_start
            skip_end = (idx != len(slices) - 1) and skip_end
            audio = infer_multilang(
                piece,
                reference_audio=reference_audio,
                emotion=emotion,
                sdp_ratio=sdp_ratio,
                noise_scale=noise_scale,
                noise_scale_w=noise_scale_w,
                length_scale=length_scale,
                sid=speaker,
                language=language[idx],
                hps=hps,
                net_g=net_g,
                device=device,
                skip_start=skip_start,
                skip_end=skip_end,
            )
            audio16bit = gr.processing_utils.convert_to_16_bit_wav(audio)
            audio_list.append(audio16bit)
            # audio_list.append(silence)  # 将静音添加到列表中
    return audio_list


def tts_split(
    text: str,
    speaker,
    sdp_ratio,
    noise_scale,
    noise_scale_w,
    length_scale,
    language,
    cut_by_sent,
    interval_between_para,
    interval_between_sent,
    reference_audio,
    emotion,
):
    if language == "mix":
        return ("invalid", None)
    while text.find("\n\n") != -1:
        text = text.replace("\n\n", "\n")
    para_list = re_matching.cut_para(text)
    audio_list = []
    if not cut_by_sent:
        for idx, p in enumerate(para_list):
            skip_start = idx != 0
            skip_end = idx != len(para_list) - 1
            audio = infer(
                p,
                reference_audio=reference_audio,
                emotion=emotion,
                sdp_ratio=sdp_ratio,
                noise_scale=noise_scale,
                noise_scale_w=noise_scale_w,
                length_scale=length_scale,
                sid=speaker,
                language=language,
                hps=hps,
                net_g=net_g,
                device=device,
                skip_start=skip_start,
                skip_end=skip_end,
            )
            audio16bit = gr.processing_utils.convert_to_16_bit_wav(audio)
            audio_list.append(audio16bit)
            silence = np.zeros((int)(44100 * interval_between_para), dtype=np.int16)
            audio_list.append(silence)
    else:
        for idx, p in enumerate(para_list):
            skip_start = idx != 0
            skip_end = idx != len(para_list) - 1
            audio_list_sent = []
            sent_list = re_matching.cut_sent(p)
            for idx, s in enumerate(sent_list):
                skip_start = (idx != 0) and skip_start
                skip_end = (idx != len(sent_list) - 1) and skip_end
                audio = infer(
                    s,
                    reference_audio=reference_audio,
                    emotion=emotion,
                    sdp_ratio=sdp_ratio,
                    noise_scale=noise_scale,
                    noise_scale_w=noise_scale_w,
                    length_scale=length_scale,
                    sid=speaker,
                    language=language,
                    hps=hps,
                    net_g=net_g,
                    device=device,
                    skip_start=skip_start,
                    skip_end=skip_end,
                )
                audio_list_sent.append(audio)
                silence = np.zeros((int)(44100 * interval_between_sent))
                audio_list_sent.append(silence)
            if (interval_between_para - interval_between_sent) > 0:
                silence = np.zeros(
                    (int)(44100 * (interval_between_para - interval_between_sent))
                )
                audio_list_sent.append(silence)
            audio16bit = gr.processing_utils.convert_to_16_bit_wav(
                np.concatenate(audio_list_sent)
            )  # 对完整句子做音量归一
            audio_list.append(audio16bit)
    audio_concat = np.concatenate(audio_list)
    return ("Success", (44100, audio_concat))


def tts_fn(
    text: str,
    speaker,
    sdp_ratio,
    noise_scale,
    noise_scale_w,
    length_scale,
    language,
    reference_audio,
    emotion,
):
    print(text, speaker, sdp_ratio, noise_scale, noise_scale_w, length_scale, language, reference_audio, emotion)
    audio_list = []
    if language == "mix":
        bool_valid, str_valid = re_matching.validate_text(text)
        if not bool_valid:
            return str_valid, (
                hps.data.sampling_rate,
                np.concatenate([np.zeros(hps.data.sampling_rate // 2)]),
            )
        result = []
        for slice in re_matching.text_matching(text):
            _speaker = slice.pop()
            temp_contant = []
            temp_lang = []
            for lang, content in slice:
                if "|" in content:
                    temp = []
                    temp_ = []
                    for i in content.split("|"):
                        if i != "":
                            temp.append([i])
                            temp_.append([lang])
                        else:
                            temp.append([])
                            temp_.append([])
                    temp_contant += temp
                    temp_lang += temp_
                else:
                    if len(temp_contant) == 0:
                        temp_contant.append([])
                        temp_lang.append([])
                    temp_contant[-1].append(content)
                    temp_lang[-1].append(lang)
            for i, j in zip(temp_lang, temp_contant):
                result.append([*zip(i, j), _speaker])
        for i, one in enumerate(result):
            skip_start = i != 0
            skip_end = i != len(result) - 1
            _speaker = one.pop()
            idx = 0
            while idx < len(one):
                text_to_generate = []
                lang_to_generate = []
                while True:
                    lang, content = one[idx]
                    temp_text = [content]
                    if len(text_to_generate) > 0:
                        text_to_generate[-1] += [temp_text.pop(0)]
                        lang_to_generate[-1] += [lang]
                    if len(temp_text) > 0:
                        text_to_generate += [[i] for i in temp_text]
                        lang_to_generate += [[lang]] * len(temp_text)
                    if idx + 1 < len(one):
                        idx += 1
                    else:
                        break
                skip_start = (idx != 0) and skip_start
                skip_end = (idx != len(one) - 1) and skip_end
                print(text_to_generate, lang_to_generate)
                audio_list.extend(
                    generate_audio_multilang(
                        text_to_generate,
                        sdp_ratio,
                        noise_scale,
                        noise_scale_w,
                        length_scale,
                        _speaker,
                        lang_to_generate,
                        reference_audio,
                        emotion,
                        skip_start,
                        skip_end,
                    )
                )
                idx += 1
    elif language.lower() == "auto":
        for idx, slice in enumerate(text.split("|")):
            if slice == "":
                continue
            skip_start = idx != 0
            skip_end = idx != len(text.split("|")) - 1
            sentences_list = split_by_language(
                slice, target_languages=["zh", "ja", "en"]
            )
            idx = 0
            while idx < len(sentences_list):
                text_to_generate = []
                lang_to_generate = []
                while True:
                    content, lang = sentences_list[idx]
                    temp_text = [content]
                    lang = lang.upper()
                    if lang == "JA":
                        lang = "JP"
                    if len(text_to_generate) > 0:
                        text_to_generate[-1] += [temp_text.pop(0)]
                        lang_to_generate[-1] += [lang]
                    if len(temp_text) > 0:
                        text_to_generate += [[i] for i in temp_text]
                        lang_to_generate += [[lang]] * len(temp_text)
                    if idx + 1 < len(sentences_list):
                        idx += 1
                    else:
                        break
                skip_start = (idx != 0) and skip_start
                skip_end = (idx != len(sentences_list) - 1) and skip_end
                print(text_to_generate, lang_to_generate)
                audio_list.extend(
                    generate_audio_multilang(
                        text_to_generate,
                        sdp_ratio,
                        noise_scale,
                        noise_scale_w,
                        length_scale,
                        speaker,
                        lang_to_generate,
                        reference_audio,
                        emotion,
                        skip_start,
                        skip_end,
                    )
                )
                idx += 1
    else:
        audio_list.extend(
            generate_audio(
                text.split("|"),
                sdp_ratio,
                noise_scale,
                noise_scale_w,
                length_scale,
                speaker,
                language,
                reference_audio,
                emotion,
            )
        )

    audio_concat = np.concatenate(audio_list)
    return "Success", (hps.data.sampling_rate, audio_concat)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", "--model", default="./logs/as/G_8000.pth", help="path of your model"
    )
    parser.add_argument(
        "-c",
        "--config",
        default="./configs/config.json",
        help="path of your config file",
    )
    parser.add_argument(
        "--share", default=False, help="make link public", action="store_true"
    )
    parser.add_argument(
        "-d", "--debug", action="store_true", help="enable DEBUG-LEVEL log"
    )

    parser.add_argument(
        "--text", help="要生成语音的文本内容"
    )
    parser.add_argument(
        "--speaker",help="角色名称，config.json中spk2id配置的那个角色名"
    )
    parser.add_argument(
        "--sdp", type=float,default=0.2, help="sdp参数，0到1之间，默认0.2"
    )
    parser.add_argument(
        "--noise", type=float,default=0.6, help="噪音规模，0.1到2之间，默认0.6"
    )
    parser.add_argument(
        "--noise_w",type=float,default=0.8, help="噪音规模w，0.1到2之间，默认0.8"
    )
    parser.add_argument(
        "--length_scale",type=float,default=1, help="长度标尺，0.1到2之间，默认1"
    )


    args = parser.parse_args()
    if config.webui_config.debug:
        logger.info("Enable DEBUG-LEVEL log")
        logging.basicConfig(level=logging.DEBUG)
    hps = utils.get_hparams_from_file(config.webui_config.config_path)
    # 若config.json中未指定版本则默认为最新版本
    version = hps.version if hasattr(hps, "version") else latest_version
    print(config.webui_config.model)
    net_g = get_net_g(
        model_path=config.webui_config.model, version=version, device=device, hps=hps
    )
    speaker_ids = hps.data.spk2id
    speakers = list(speaker_ids.keys())
    languages = ["ZH", "JP", "EN", "mix", "auto"]

    print(f'运行参数：{args}')
    #调用tts：
    msg,audios = tts_fn(
                text=args.text,
                speaker=args.speaker,
                sdp_ratio=args.sdp,
                noise_scale=args.noise,
                noise_scale_w=args.noise_w,
                length_scale=args.length_scale,
                language='ZH',
                reference_audio=None,
                emotion=0,
                )
    if msg=='Success':
          sound = AudioSegment(
            audios[1].tobytes(),
            frame_rate=audios[0],
            sample_width=2,
            channels=1
          )
          #存为wav
          sound.export(f"音频-{args.text}.wav", format="wav")

    print(f'生成音频完成！')

    '''
    命令行调用：
        python 命令行推理.py -m 模型所在路径.pth \
            --text 你好啊你是谁呀 \
            --speaker 角色名
            
    '''
