import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import traceback

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from data.data_handler import AudioHandler, DataInfoHandler
from data.data_config import DataConfig
from utils.util_audio import SoxEffectTransform

import torch
import random
import numpy as np


class AudioManager():
    def __init__(self, args):
        hparam_dict = vars(args)
        self.config = DataConfig()
        self.config.set_params(hparam_dict)
        print(args.MODEL_INFO)
        if args.MODEL_INFO == 'CPC':
            self.config.params['IS_FULL_AUDIO'] = True
            self.config.params['IS_POS_SAMPLING'] = False

        print('-- Audio Loader Parameters --')
        print(self.config.params)

        self.info_handler = DataInfoHandler(self.config)
        self.audio_handler = AudioHandler(self.config)
        self.dataset = AudioTorchDataset(self.config, self.info_handler, self.audio_handler)
        self.dataloader = DataLoader(self.dataset, 
                                     batch_size=self.config.params['BATCH_SIZE'],
                                     shuffle=True,
                                     drop_last=False,
                                     collate_fn=self.collate_fn,
                                     num_workers=self.config.params['NUM_WORKERS']
                                     )

    def collate_fn(self, batch):
        batch = list(filter(lambda x: x is not None, batch))
        return torch.utils.data.dataloader.default_collate(batch)


class AudioTorchDataset(Dataset):
    def __init__(self, config, info_handler, audio_handler):
        self.config = config
        self.info_handler = info_handler
        self.audio_handler = audio_handler
    
    def __getitem__(self, idx):
        """
        - Get idx-th file path from info_handler
        - Do loading / preprocessing from audio_handler
        - Do augmentation
        """

        curr_tr_id = self.info_handler.tr_id_list[idx]
        curr_tr_path = self.info_handler.tr_file_path_list[idx]
        try:
            # 1. Positive sampling (returning 2 segments)
            if self.config.params['IS_POS_SAMPLING']:

                # (1) Loading
                if self.config.params['POS_STRATEGY'] == 'self':  # self
                    anc_wav, _ = self.audio_handler.load_single_wav_tensor(curr_tr_path, is_full=self.config.params['IS_FULL_AUDIO'])
                    pos_wav = torch.empty_like(anc_wav).copy_(anc_wav)

                else:  # adjacent / random / random-wo-repl
                    curr_wavs, _ = self.audio_handler.load_two_wav_tensors(curr_tr_path, self.config.params['POS_STRATEGY'])
                    anc_wav, pos_wav = curr_wavs[0], curr_wavs[1]

                # (2) Augmentation
                if self.config.params['ANCHOR_WAV_AUG']:
                    pitch_change = random.randint(100 * self.config.params['PS_LOW'], 100 * self.config.params['PS_HIGH'])
                    speed_factor = np.random.uniform(low=self.config.params['TS_LOW'], high=self.config.params['TS_HIGH'])
                    augmentation = [
                        ['remix', '-'],
                        ['pitch', f'{pitch_change}'],
                        ['speed', f'{speed_factor:.5f}'],
                    ]
                    anc_transform = SoxEffectTransform(augmentation)
                    anc_wav, _ = anc_transform(anc_wav, self.config.params['SR'])

                if self.config.params['POS_WAV_AUG']:
                    pitch_change = random.randint(100 * self.config.params['PS_LOW'], 100 * self.config.params['PS_HIGH'])
                    speed_factor = np.random.uniform(low=self.config.params['TS_LOW'], high=self.config.params['TS_HIGH'])
                    augmentation = [
                        ['remix', '-'],
                        ['pitch', f'{pitch_change}'],
                        ['speed', f'{speed_factor:.5f}'],
                    ]
                    pos_transform = SoxEffectTransform(augmentation)
                    pos_wav, _ = pos_transform(pos_wav, self.config.params['SR'])

                if self.config.params['IS_FULL_AUDIO']:
                    anc_wav = self.audio_handler.torch_wav_zero_pad(anc_wav, self.config.params['FULL_LENGTH_WAV_FRAME'])
                    pos_wav = self.audio_handler.torch_wav_zero_pad(pos_wav, self.config.params['FULL_LENGTH_WAV_FRAME'])
                else:
                    anc_wav = self.audio_handler.torch_wav_zero_pad(anc_wav, self.config.params['SEGMENT_LENGTH_WAV_FRAME'])
                    pos_wav = self.audio_handler.torch_wav_zero_pad(pos_wav, self.config.params['SEGMENT_LENGTH_WAV_FRAME'])

                # (3) Transformation
                if self.config.params['AUDIO_REPR'] == 'wav':
                    return (anc_wav, pos_wav), curr_tr_id
                
                else:
                    anc, pos = self.audio_handler.repr_transform(anc_wav).squeeze(), self.audio_handler.repr_transform(pos_wav).squeeze()
                    if self.config.params['AUDIO_REPR'] == 'mel':
                        anc -= self.config.params['LOG_MEL_TRAIN_MEAN-' + self.config.params['CURR_DATASET']]
                        anc /= self.config.params['LOG_MEL_TRAIN_STD-' + self.config.params['CURR_DATASET']]
                        pos -= self.config.params['LOG_MEL_TRAIN_MEAN-' + self.config.params['CURR_DATASET']]
                        pos /= self.config.params['LOG_MEL_TRAIN_STD-' + self.config.params['CURR_DATASET']]

                    anc = torch.transpose(anc, 0, 1)
                    pos = torch.transpose(pos, 0, 1)

                    assert anc.size() == pos.size()
                    return (anc, pos), curr_tr_id


            # 2. Sampling (returning 1 segment.)
            else:
                anc_wav, _ = self.audio_handler.load_single_wav_tensor(curr_tr_path, 
                                                                    is_full=self.config.params['IS_FULL_AUDIO'])
                # (1) Augmentation 
                if self.config.params['ANCHOR_WAV_AUG']:
                    pitch_change = random.randint(100 * self.config.params['PS_LOW'], 100 * self.config.params['PS_HIGH'])
                    speed_factor = np.random.uniform(low=self.config.params['TS_LOW'], high=self.config.params['TS_HIGH'])
                    augmentation = [
                        ['remix', '-'],
                        ['pitch', f'{pitch_change}'],
                        ['speed', f'{speed_factor:.5f}'],
                    ]
                    anc_transform = SoxEffectTransform(augmentation)
                    anc_wav, _ = anc_transform(anc_wav, self.config.params['SR'])
                
                if self.config.params['IS_FULL_AUDIO']:
                    anc_wav = self.audio_handler.torch_wav_zero_pad(anc_wav, self.config.params['FULL_LENGTH_WAV_FRAME'])
                else:
                    anc_wav = self.audio_handler.torch_wav_zero_pad(anc_wav, self.config.params['SEGMENT_LENGTH_WAV_FRAME'])

                # (2) Transformation
                if self.config.params['AUDIO_REPR'] == 'wav':
                    return anc_wav, curr_tr_id
                
                else:
                    anc = self.audio_handler.repr_transform(anc_wav).squeeze()
                    if self.config.params['AUDIO_REPR'] == 'mel':
                        anc -= self.config.params['LOG_MEL_TRAIN_MEAN-' + self.config.params['CURR_DATASET']]
                        anc /= self.config.params['LOG_MEL_TRAIN_STD-' + self.config.params['CURR_DATASET']]
                    anc = torch.transpose(anc, 0, 1)
                    return anc, curr_tr_id
        except Exception as e:
            print("".join(traceback.format_tb(e.__traceback__)))
            return None

    def __len__(self):
        return len(self.info_handler.tr_id_list)
