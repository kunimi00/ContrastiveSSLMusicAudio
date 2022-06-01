import os
import yaml

class DataConfig(object):
    def __init__(self):
        with open('data/data_config.yaml', 'r') as f:
            self.params = yaml.load(f, Loader=yaml.FullLoader)
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

        if self.params['MAX_AUDIO_LENGTH_SEC'] is not None:
            self.params['FULL_LENGTH_WAV_FRAME'] = int(self.params['SR'] * self.params['MAX_AUDIO_LENGTH_SEC'])
        else:
            self.params['FULL_LENGTH_WAV_FRAME'] = 0


        if self.params['GPU'] is None:
            self.params['DEVICE'] = "cpu"
        else:
            self.params['DEVICE' ] = "cuda:" + str(self.params['GPU'])

    def set_params(self, param_dict):
        for key in param_dict:
            if key in self.params:
                self.params[key] = param_dict[key]
        self.set_conditional_params()
