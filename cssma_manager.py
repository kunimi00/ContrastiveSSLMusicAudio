import sys
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from model.audio_encoders import *
from model.emb_aggregators import *
from model.contrastive_models import *

class ContrastiveSelfSupervisionManager:
    def __init__(self, config):
        self.config = config

        if config.AUDIO_ENCODER == 'MelConv5by5Enc':
            self.audio_encoder = MelConv5by5Enc(config)
        elif config.AUDIO_ENCODER == 'MelConv5by5EncMulti':
            self.audio_encoder = MelConv5by5EncMulti(config)
        elif config.AUDIO_ENCODER == 'SampleCNN':
            pass
        
        if config.MODEL_INFO == 'Siamese':
            self.contrastive_model = ModelSiamese(config, self.audio_encoder)
        elif config.MODEL_INFO == 'CPC':
            if config.AUDIO_ENCODER == 'MelConv5by5Enc':
                self.emb_aggregator = CPCModule(config)
                self.contrastive_model = ModelCPC(config, self.audio_encoder, self.emb_aggregator)
            elif self.config.AUDIO_ENCODER == 'MelConv5by5EncMulti':
                self.emb_aggregator_list = []
                for i in range(5):
                    emb_aggregator = CPCModule(config)
                    self.emb_aggregator_list.append(emb_aggregator)
                self.contrastive_model = ModelCPC(config, self.audio_encoder, self.emb_aggregator_list)
                
        elif config.MODEL_INFO == 'Transformer':
            pass

    
    def train_model(self, dataloader):
        self.contrastive_model.train(dataloader, epoch_from=self.config.EPOCH_FROM)
        print('Training done.')


    def inference(self, dataloader):
        self.contrastive_model.predict(dataloader)
        print('Inference done.')
    
    