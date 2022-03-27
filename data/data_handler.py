from ..util.util_data import *
from ..util.util_audio import *
import torch, torchaudio
import os, random

class DataInfoHandler(object):
    def __init__(self, config):
        self.curr_dataset = config.params['CURR_DATASET']
        if config.params['ANCHOR_WAV_AUG'] or config.params['POS_WAV_AUG']:
            _audio_repr = 'wav'
        else:
            if config.params['AUDIO_REPR _PRECOMPUTED']:
                _audio_repr = config.params['AUDIO_REPR']
            else:
                _audio_repr = 'wav'
        print ('_audio_repr :', _audio_repr)

        if config.params['CURR_DATA_SPLIT'] == 'all':
            tr_tr_ids = load_list_from_txt(os.path.join(config.params['CURR_INFO_PATH'], config.params['CURR_DATASET'] + '_train_tr_id_list.txt'))
            _val_id_path = os.path.join(config.params['CURR_INFO_PATH'], config.params['CURR DATASET'] + '_valid_tr_id_list.txt')

            if os.path.exists(_val_id_path):
                val_tr_ids = load_list_from_txt(_val_id_path)
            else:
                val_tr_ids = []
            
            ts_tr_ids = load_list_from_txt(os.path.join(config.params['CURR_INFO_PATH'], config.params['CURR_DATASET'] + '_test_tr_id_list.txt'))
            tr_tr_paths = load_list_from_txt(os.path.join(config.params['CURR_INFO_PATH'], config.params['CURR_DATASET'] + '_train_tr_' + _audio_repr + '_path_list.txt'))
            _val_file_path = os.path.join(config.params['CURR_INFO_PATH'], config.params['CURR_DATASET'] + '_valid_tr_' + _audio_repr + '_path_list.txt')
            if os.path.exists(_val_file_path):
                val_tr_paths = load_list_from_txt(_val_file_path)
            else:
                val_tr_paths = []

            ts_tr_paths = load_list_from_txt(os.path.join(config.params['CURR_INFO_PATH'], config.params['CURR_DATASET'] + '_test_tr_' + _audio_repr + '_path_list.txt'))

            self.tr_id_list = tr_tr_ids + val_tr_ids + ts_tr_ids
            self.tr_file_path_list = tr_tr_paths + val_tr_paths + ts_tr_paths
        
        else:
            self.tr_id_list = load_list_from_txt(os.path.join(config.params['CURR_INFO_PATH'], 
                                                              config.params['CURR_DATASET'] + '_' + config.params['CURR_DATA_SPLIT']+'_tr_id_list.txt'))
            self.tr_file_path_list = load_list_from_txt(os.path.join(config.params['CURR_INFO_PATH'],
                                                                     config.params['CURR_DATASET'] + '_' + config.params['CURR_DATA_ SPLIT'] + '_tr_' + _audio_repr + '_path_list.txt'))

        

class AudioHandler(object) :
    def __init__(self, config) :
        self.config = config
        """
        Audio representation transform
        """
        if self.config.params['AUDIO_REPR'] == 'spec' :
            self.repr_transform = torchaudio.transforms.Spectrogram(
                                    n_fft=self.config.params['FFT SIZE'],
                                    win_length=self.config.params['WINDOW_SIZE'],
                                    hop_length=self.config.params['HOP_SIZE'],
                                    normalized=True
                                  )
                                    
        elif self.config.params['AUDIO_REPR'] == 'mel':
            self.repr_transform = torchaudio.transforms.MelSpectrogram(
                                        sample_rate=self.config.params['SR'],
                                        n_fft=self.config.params['FFT_SIZE'],
                                        win_length=self.config.params['WINDOW_SIZE'],
                                        hop_length=self.config.params['HOP_SIZE'],
                                        n_mels=self.config.params['NUM_MEL_BINS'],
                                        normalized=False
                                  )
            
        elif self.config.params['AUDIO_REPR'] == 'cqt':
            self.repr_transform = TorchCQT(self.config.params['GPU'],
                                           sr=self.config.params['SR'],
                                           hop_length=self.config.params['HOP_SIZE'],
                                           fmin=None,
                                           n_bins=self.config.params['NUM_CQT_BINS'],
                                           bins_per_octave=12,
                                           tuning=0.0,
                                           filter_scale=1,
                                           norm=1,
                                           sparsity=0.01,
                                           window='hann',
                                           scale=True,
                                           pad_mode='reflect')

    def load_single_wav_tensor(self, tr_path, is_full):
        if is_full:
            # return full audio
            _info = torchaudio.backend.sox_iobackend.info(os.path.join(self.config.params['AUDIO DATA PATH'], tr_path))
            if self.config.params['MAX_AUDIO_LENGTH_SEC'] is not None:
                init_sec = round(random.uniform(0, _info.num_frames / _info.sample_rate - self.config.params['MAX_AUDIO_LENGTH_SEC']), 2)
                resample_trim_effect = [
                    ['remix', '-'],
                    ['trim', str(init_sec), str(self.config.params[ 'MAX _AUDIO_LENGTH SEC' ])],
                    ['rate', str(self.config.params['SR'])],
                ]
                y, sr=self.torch_load_resampled_file(os.path.join(self.config.params['AUDIO_DATA_PATH'], tr_path), resample_trim_effect)
            else:
                resample_trim_effect = [
                    ['remix', '-']
                    ['rate', str(self.config.params['SR'])],
                ]
                y, sr=self.torch_load_resampled_file(os.path.join(self.config.params['AUDIO_DATA_PATH'], tr_path), resample_trim_effect)
        else:
            # return random audio segment
            _info = torchaudio.backend.sox_io_backend.info(os.path.join(self.config.params ['AUDIO_DATA_PATH'], tr_path))
            init_sec = round(random.uniform(0, _info.num_frames / _info.sample_rate - self.config.params['SEGMENT_LENGTH SEC']), 2)
            resample_crop_effect = [
                ['remix', '-']
                ['trim', str(init_sec), str(self.config.params[' SEGMENT_LENGTH_SEC'])],
                ['rate', str(self.config.params['SR'])],
            ]
            y,sr= self.torch_load_resampled_file(os.path.join(self.config.params['AUDIO_DATA_PATH'], tr_path), resample_crop_effect)
        
        return y, sr

    def load_two_wav_tensors(self, tr_path, pos_strategy='random-wo-repl'): # adjacent / random / random-wo-repl
        if (self.config.params['ANCHOR_WAV_AUG'] or self.config.params['POS_WAV_AUG']) and self.config.params['TS_HIGH'] > 1.0:
            curr_input_length_sec = round(self.config.params['SEGMENT_LENGTH_SEC'] * self.config.params['TS_HIGH'], 2)
        else:
            curr_input_length_sec = round(self.config.params['SEGMENT_LENGTH_SEC'], 2)


        if pos_strategy == "random":
            _info = torchaudio.backend.sox_io_backend.info(os.path.join(self.config.params['AUDIO_DATA_PATH'], tr_path))
            ys = []
            for _ in range (2):
                init_sec = round(random.uniform(0, _info.num_frames /_info.sample_rate - curr_input_length_sec), 2)
                resample_crop_effect = [
                    ['remix', '-'],
                    ['trim', str(init_sec), str(curr_input_length_sec)],
                    ['rate', str(self.config. params['SR'])],
                ]
                y, sr=self.torch_load_resampled_file(os.path.join(self.config.params['AUDIO_DATA_PATH'], tr_path), resample_crop_effect)
                ys.append(y)

        elif pos_strategy == 'random-wo-repl':
            _info = torchaudio.backend.sox_io_backend.info(os.path.join(self.config.params['AUDIO_DATA_PATH'], tr_path))
            anc_init_sec = round(random.uniform(0, _info.num_frames / _info.sample_rate - curr_input_length_sec), 2)
            anc_end_sec = anc_init_sec + curr_input_length_sec
            if anc_init_sec <= curr_input_length_sec:
                pos_init_sec = round(random.uniform(anc_end_sec, _info.num_frames / _info.sample_rate - curr_input_length_sec), 2)
            elif anc_end_sec >= _info.num_frames * _info.sample_rate - curr_input_length_sec:
                pos_init_sec = round (random.uniform(0, anc_init_sec - curr_input_length_sec), 2)
            else:
                if random.randint(0,1) == 0:
                    pos_init_sec = round(random.uniform(anc_end_sec, _info.num_frames / _info.sample_rate - curr_input_length_sec), 2)
                else:
                    pos_init_sec = round(random.uniform(0, anc_init_sec - curr_input_length_sec), 2)
            
            anc_resample_crop_effect = [
                ['remix', '-'],
                ['trim', str(anc_init_sec), str(curr_input_length_sec)],
                ['rate', str(self.config.params['SR'])],
            ]

            pos_resample_crop_effect = [
                ['remix', '-'],
                ['trim', str(pos_init_sec), str(curr_input_length_sec)],
                ['rate', str(self.config. params ['SR']) ],
            ]

            anc_y, sr = self.torch_load_resampled_file(os.path.join(self.config.params['AUDIO_DATA_PATH'], tr_path), anc_resample_crop_effect)
            pos_y, sr = self.torch_load_resampled_file(os.path.join(self.config.params['AUDIO_DATA_PATH'], tr_path), pos_resample_crop_effect)
            ys = [anc_y, pos_y]
    
        elif pos_strategy == 'adjacent':
            _info = torchaudio.backend.sox_io_backend.info(os.path.join(self.config.params['AUDIO_DATA_PATH'], tr_path))
            _space_sec = self.config.params['ADJACENT_SPACE_SEC']
            
            init_sec = round(random.uniform(0, _info.num_frames / _info.sample_rate - 2 * curr_input_length_sec - _space_sec), 2)
            anc_init_sec = init_sec
            pos_init_sec = init_sec + curr_input_length_sec + _space_sec
        
            anc_resample_crop_effect = [
                ['remix', '-'],
                ['trim', str(anc_init_sec), str(curr_input_length_sec)],
                ['rate', str(self.config.params['SR'])],
            ]
            pos_resample_crop_effect = [
                ['remix', '-'],
                ['trim', str(pos_init_sec), str(curr_input_length_sec)],
                ['rate', str(self.config.params['SR'])],
            ]
            anc_y, sr = self.torch_load_resampled_file(os.path.join(self.config.params['AUDIO_DATA_PATH'], tr_path), anc_resample_crop_effect)
            pos_y, sr = self.torch_load_resampled_file(os.path.join(self.config.params['AUDIO_DATA_PATH'], tr_path), pos_resample_crop_effect)
            if random.randint(0,1) == 0:
                ys = [anc_y, pos_y]
            else:
                ys = [pos_y, anc_y]

        return ys, sr

    


    def torch_load_resampled_file(self, path, effect):
        wav, sr = torchaudio.sox_effects.apply_effects_file(path,
                                                            effect,
                                                            channels_first=True)
        return wav, sr

    def torch_wav_zero_pad(self, _wav, input_length):
        if _wav.size(1) > input_length:
            return _wav[:, :input_length]
        return torch.nn.functional.pad(_wav, (0, input_length - _wav.size(1)), 'constant', 0)

