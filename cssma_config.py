import sys
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import yaml
from pathlib import Path
from util.util_path import *

class ContrastiveSelfSupervisionConfig(object):
    def __init__(self, args):
        self.__dict__.update(vars(args))
        self.set_conditional_params()


    def set_conditional_params(self):
        """
        Weight / inferenced embedding folder structure

            wav or mp3 files : data_path/curr_dataset_name/wav
            mel or cqt files : data_path/curr_dataset_name/mel or cqt
            track ids and paths : data_path/curr_dataset_name/data
        """
        
        """
        Experiment information 
        """
        if self.LOAD_WEIGHT_FROM is None:
            self.CURR_WEIGHT_FILE = None
            self.EPOCH_FROM = 0
            self.CURR_WEIGHT_FOLDER = create_weight_folder_name(self)

        else: 
            """
            Continue learning or inference from saved weight
            """
            parsed = parse_weight_path(self.LOAD_WEIGHT_FROM)
            self.EXP_INFO = parsed[0]
            self.MODEL_INFO = parsed[1]  
            self.AUDIO_REPR = parsed[2]
            self.AUDIO_ENCODER = parsed[3]  
            self.POS_STRATEGY = parsed[4]  # 'random' / 'adjacent' / 'self' / 'random-wo-repl' 
            self.NEG_STRATEGY = parsed[5]  # 'inter-track' / 'intra-track'
            self.POS_WAV_AUG = parsed[6]
            self.EMB_DIM = parsed[7]

            self.CURR_WEIGHT_FILE = parsed[8]
            self.EPOCH_FROM = parsed[9]
            self.CURR_WEIGHT_FOLDER = self.LOAD_WEIGHT_FROM.split('/')[-2]
    
        """
        Weight, Inference paths
        """
        self.WEIGHT_PATH = os.path.join(self.SAVE_DIR_PATH, 'checkpoints')
        self.INFERENCE_PATH = os.path.join(self.SAVE_DIR_PATH, 'inferenced')
        Path(os.path.dirname(self.WEIGHT_PATH)).mkdir(parents=True, exist_ok=True)
        Path(os.path.dirname(self.INFERENCE_PATH)).mkdir(parents=True, exist_ok=True)
      
        """
        Model parameters
        """

        if self.MODEL_INFO == 'CPC' or self.MODEL_INFO == 'Transformer':
            self.NUM_TIMESTEPS = self.MAX_AUDIO_LENGTH_SEC // self.SEGMENT_LENGTH_SEC
        
        else:  # Siamese Network
            if self.NEG_STRATEGY == 'inter-track':
                self.NUM_TIMESTEPS = 1
            else:
                self.NUM_TIMESTEPS = self.MAX_AUDIO_LENGTH_SEC // self.SEGMENT_LENGTH_SEC + 1
            
        """
        Device
        """
        if self.GPU is None:
            self.DEVICE = "cpu"
        else:
            self.DEVICE = "cuda:" + str(self.GPU)

        