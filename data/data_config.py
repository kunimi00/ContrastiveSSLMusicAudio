import os

class DataConfig(object):
    def __init__(self):
        self.params = {
            'SVR' : 's03',
            'CURR_DATASET' : 'MSD',
            'CURR_DATA_SPLIT' : 'train',  # 'all' / 'train' / 'valid' / 'test'
            'AUDIO_REPR' : 'mel',  # 'spec' / 'mel' / 'wav' (mp3) / 'cqt'
            'FULL_DATA_PATH' : '/mnt/nas_nlp/Knowledge_AI_Lab/reasoning/jchoi/data/',
            'AUDIO_REPR_PRECOMPUTED' : False,
            'SR' : 16000,
            'FFT_SIZE' : 512,
            'WINDOW_SIZE' : 512,
            'HOP_SIZE' : 256,
            'NUM_SPEC_BINS' : 128,
            'NUM_MEL_BINS' : 96,
            'NUM_CQT_BINS' : 84,
            'SEGMENT_LENGTH_SEC' : 3,
                'ANCHOR_WAV_AUG' : False,
                'ANCHOR_SPEC_AUG' : False,
            'IS_POS_SAMPLING' : True,
                'POS_STRATEGY' : 'random',  # 'self' / 'adjacent' / 'random' / 'random-wo-repl'
                'POS_WAV_AUG' : True,
                'POS_SPEC_AUG' : True,
                'ADJECENT_SPACE_SEC' : 0.1,
            'IS_FULL_AUDIO' : False,
            'MAX_AUDIO_LENGTH_SEC' : 29,
            'TS_LOW' : 0.8,
            'TS_HIGH' : 1.2,
            'PS_LOW' : -2,  # in semitones
            'PS_HiGH' : 2,
            'LOG_MEL_TRAIN_MEAN-MSD' : 0.1533010751863198,
            'LOG_MEL_TRAIN_STD-MSD' : 0.17129572356836764,
            'LOG_MEL_TRAIN_MEAN-FMA_small' : 0.1325546,
            'LOG_MEL_TRAIN_STD-FMA_small' : 0.18435284,
            'LOG_MEL_TRAIN_MEAN-IRMAS' : 0.1533010751863198,
            'LOG_MEL_TRAIN_STD-IRMAS' : 0.17129572356836764,
            'LOG_MEL_TRAIN_MEAN-SALAMI' : 0.08593344,
            'LOG_MEL_TRAIN_STD-SALAMI' : 0.13943474,
            'LOG_MEL_TRAIN_MEAN-BEATLES' : 0.13648747,
            'LOG_MEL_TRAIN_STD-BEATLES' : 0.16205402,
            'GPU' : None,
            'BATCH_SIZE' : 16,
            'NUM_WORKERS' : 16

        }
        self.set_conditional_params()

    def set_conditional_params(self):
        if self.params['AUDIO_REPR_PRECOMPUTED']:
            if self.params['CURR_DATASET'] == 'AOTM':
                self.params['AUDIO_DATA_PATH'] = os.path.join(self.params['FULL_DATA_PATH'], 'MSD', self.params['AUDIO_REPR'])
            else:
                self.params['AUDIO_DATA_PATH'] = os.path.join(self.params['FULL_DATA_PATH'], 
                                                            self.params['CURR_DATASET'], 
                                                            self.params['AUDIO_REPR'])
        else:
            if self.params['CURR_DATASET'] == 'AOTM':
                self.params['AUDIO_DATA_PATH'] = os.path.join(self.params['FULL_DATA_PATH'], 'MSD', 'wav')
            else:
                self.params['AUDIO_DATA_PATH'] = os.path.join(self.params['FULL_DATA_PATH'], 
                                                            self.params['CURR_DATASET'], 
                                                            'wav')
        
        self.params['CURR_INFO_PATH'] = os.path.join(self.params['FULL_DATA_PATH'],
                                                     self.params['CURR_DATASET'],
                                                     'info')
        self.params['SEGMENT_LENGTH_WAV_FRAME'] = int(self.params['SR'] * self.params['SEGMENT_LENGTH_SEC'])

        if self.params['MAX _AUDIO_LENGTH_SEC'] is not None:
            self.params['FULL_LENGTH _WAV_FRAME'] = int(self.params['SR'] * self.params['MAX AUDIO LENGTH_SEC'])
        else:
            self.params['FULL LENGTH WAV FRAME'] = 0


        if self.params['GPU' ] is None:
            self.params['DEVICE'] = "cpu"
        else:
            self.params['DEVICE' ] = "cuda:" + str(self.params['GPU'])

    def set_params(self, param_dict):
        for key in param_dict:
            self.params[key] = param_dict[key]
            self. set_conditional_params()
