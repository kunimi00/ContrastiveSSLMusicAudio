"""
2021/05/27 Jeong Choi

모델 관련 Configuration 클래스
- hparams를 넘겨 받아서 모델 학습에 필요한 parameter 지정.
- 데이터 관련 디폴트 값
    - Training 데이터 : MSD Training set
    - Inference 데이터 : FMA_small all, MSD all, IRMAS all, AOTM all
    - data_path : '/mnt/nas_nlp/Knowledge_AI_Lab/reasoning/jchoi/data/' 

"""
import os
import yaml
from .util.util_path import *

class SelfSupervisedMusicAudioConfig(object):
    def __init__(self, hparams):
        """
        Training parameters

        """
        self.SVR = hparams.svr
        self.VERBOSE = hparams.verbose
        self.IS_INFERENCE = hparams.is_inference
        
        self.TRAIN_DATASET = hparams.train_data  # 'MSD' / 'FMA_small' / 'IRMAS' / 'BEATLES' / 'AOTM'
        self.TRAIN_DATA_SPLIT = hparams.train_data_split  #  'all' / 'train' / 'valid' / 'test'
        self.INFERENCE_DATA = hparams.inference_data  # 'MSD' / 'FMA_small' / 'IRMAS' / 'BEATLES' / 'AOTM'
        self.INFERENCE_DATA_SPLIT = hparams.inference_data_split


        """
        Weight / inferenced embedding folder structure

            wav or mp3 files : data_path/curr_dataset_name/wav
            mel or cqt files : data_path/curr_dataset_name/mel or cqt
            track ids and paths : data_path/curr_dataset_name/data
        """
        self.SAVE_DIR_PATH = hparams.save_dir_path
        self.WEIGHT_PATH = os.path.join(self.SAVE_DIR_PATH, 'checkpoints', 'ssma')
        self.INFERENCE_PATH = os.path.join(self.SAVE_DIR_PATH, 'inferenced', 'ssma')

        self.LOAD_WEIGHT_FROM = hparams.load_weight_from
      
        """
        Experiment information 
        """
        if self.LOAD_WEIGHT_FROM is None:
            self.EXP_INFO = hparams.exp_info
            self.MODEL_INFO = hparams.model_info  # 'Siamese' / 'CPC' / 'MelTransformer'
            self.AUDIO_REPR = hparams.audio_repr
            self.AUDIO_ENCODER = hparams.audio_encoder  
            self.LOSS_FN = hparams.loss_fn
            self.POS_STRATEGY = hparams.pos_strategy  # 'rand' / 'adjacent' / 'self'
            self.NEG_STRATEGY = hparams.neg_strategy  # 'inter-track' / 'intra-track'
            self.POS_WAV_AUG = hparams.pos_wav_aug
            self.ANCHOR_WAV_AUG = hparams.anchor_wav_aug
            self.POS_SPEC_AUG = hparams.pos_spec_aug
            self.ANCHOR_SPEC_AUG = hparams.anchor_spec_aug
            self.EMB_DIM = hparams.emb_dim

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
            self.LOSS_FN = parsed[4]
            self.POS_STRATEGY = parsed[5]  # 'rand' / 'adjacent' / 'self'
            self.NEG_STRATEGY = parsed[6]  # 'inter-track' / 'intra-track'
            self.POS_WAV_AUG = parsed[7]
            self.EMB_DIM = parsed[8]

            self.CURR_WEIGHT_FILE = parsed[9]
            self.EPOCH_FROM = parsed[10]

            self.CURR_WEIGHT_FOLDER = self.LOAD_WEIGHT_FROM.split('/')[-2]
        
        
        """
        Model parameters
        """
        self.TRIPLET_MARGIN_DIST = hparams.triplet_margin_dist  # 'l2'
        self.TRIPLET_LOSS_MARGIN = hparams.triplet_loss_margin  # 1.

        self.INFONCE_TAO = hparams.infonce_tao  # 0.05 / 1.
        
        self.MAX_AUDIO_LENGTH_SEC = hparams.max_audio_length_sec
        self.SEGMENT_LENGTH_SEC = hparams.segment_length_sec
        
        self.NUM_CPC_INPUT_STEPS = hparams.num_cpc_input_steps
        self.NUM_CPC_PREDICTIONS = hparams.num_cpc_predictions

        if self.MODEL_INFO == 'CPC' or self.MODEL_INFO == 'Transformer':
            self.NUM_TIMESTEPS = self.MAX_AUDIO_LENGTH_SEC // self.SEGMENT_LENGTH_SEC
        
        else:  # Siamese Network
            if self.NEG_STRATEGY == 'inter-track':
                self.NUM_TIMESTEPS = 1
            else:
                self.NUM_TIMESTEPS = self.MAX_AUDIO_LENGTH_SEC // self.SEGMENT_LENGTH_SEC + 1
        
        """
        Transformer hyperparameters
        """
        
        if self.MODEL_INFO == 'Transformer':
            self.TRANSFORMER_CONFIG_YAML_PATH = 'ssma/model_config/architectures.yaml'
            self.TRANSFORMER_CONFIG_TYPE = hparams.transformer_config_type  # 'B'
            self.TRANSFORMER_NUM_MASKS = hparams.transformer_num_masks
        
            with open(self.TRANSFORMER_CONFIG_YAML_PATH) as f:
                parsed_configs = yaml.load(f, Loader=yaml.FullLoader)
            self.TRANSFORMER_CONFIG = parsed_configs[self.TRANSFORMER_CONFIG_TYPE]
            self.IS_LEARNABLE_PE = hparams.is_learnable_pe
            self.CLS_TKN = 0
            self.MASK_TKN = 1

        
        """
        Training parameters
        """
        
        self.GPU = hparams.gpu  # 0
        if self.GPU is None:
            self.DEVICE = "cpu"
        else:
            self.DEVICE = "cuda:" + str(hparams.gpu)

        self.BATCH_SIZE = hparams.batch_size  # 16
        self.NUM_EPOCHS = hparams.num_epochs  # 2000
        self.NUM_WORKERS = hparams.num_workers  # 4
        self.LR = hparams.lr  # 1e-3
        self.STOPPING_LR = hparams.stopping_lr  # 1e-10
        self.MOMENTUM = hparams.momentum  #0.9
        self.LR_FACTOR = hparams.lr_factor  # 0.2
        self.LR_PATIENCE = hparams.lr_patience  # 5
        self.SAVE_PER_EPOCH = hparams.save_per_epoch  # 5
        self.LOSS_CHECK_PER = hparams.loss_check_per

