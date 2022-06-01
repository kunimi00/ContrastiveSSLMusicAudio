import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import yaml

from cssma_manager import ContrastiveSelfSupervisionManager
from cssma_config import ContrastiveSelfSupervisionConfig
from data.data_manager import AudioManager

def run(config):
    audio_manager = AudioManager(config)

    train_config = ContrastiveSelfSupervisionConfig(config)
    ccsma_manager = ContrastiveSelfSupervisionManager(train_config)
    
    if config.MODE == 'train':
        ccsma_manager.train_model(audio_manager.dataloader)
    else:
        ccsma_manager.inference(audio_manager.dataloader)
    print('done.')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--USE_YAML_CONFIG', type=int, default=None)

    parser.add_argument('--VERBOSE', action='store_true')
    parser.add_argument('--MODE', type=str, default='train', choices=['train', 'inference'])

    parser.add_argument('--CURR_DATASET', type=str, default='FMA_small', choices=['MSD', 'FMA_small', 'IRMAS', 'BEATLES', 'AOTM'])
    parser.add_argument('--CURR_DATA_SPLIT', type=str, default='train', choices=['all', 'train', 'valid', 'test'])
    parser.add_argument('--FULL_DATA_PATH', type=str, default='/data_ssd/audio_dev/tagging/cssma_data/')

    parser.add_argument('--EXP_INFO', type=str, default='my-fma-exp')
    parser.add_argument('--MODEL_INFO', type=str, default='Siamese', choices=['Siamese', 'CPC'])
    parser.add_argument('--AUDIO_REPR', type=str, default='mel', choices=['mel'])
    parser.add_argument('--AUDIO_ENCODER', type=str, default='MelConv5by5Enc', choices=['MelConv5by5Enc', 'MelConv5by5EncMulti'])
    
    parser.add_argument('--POS_STRATEGY', type=str, default='random', choices=['random', 'adjacent', 'self', 'random-wo-repl'])
    parser.add_argument('--NEG_STRATEGY', type=str, default='inter-track', choices=['inter-track', 'intra-track'])
    parser.add_argument('--POS_WAV_AUG', action='store_true')
    parser.add_argument('--ANCHOR_WAV_AUG', action='store_true')
    parser.add_argument('--POS_SPEC_AUG', action='store_true')
    parser.add_argument('--ANCHOR_SPEC_AUG', action='store_true')
    parser.add_argument('--EMB_DIM', type=int, default=128)

    parser.add_argument('--EPOCH_FROM', type=int, default=0)
    parser.add_argument('--INFONCE_TAO', type=float, default=0.05)
    parser.add_argument('--MAX_AUDIO_LENGTH_SEC', type=float, default=29.)
    parser.add_argument('--SEGMENT_LENGTH_SEC', type=float, default=3.)
    parser.add_argument('--ADJACENT_SPACE_SEC', type=float, default=0.5)

    parser.add_argument('--NUM_CPC_INPUT_STEPS', type=int, default=4)
    parser.add_argument('--NUM_CPC_PREDICTIONS', type=int, default=4)

    parser.add_argument('--BATCH_SIZE', type=int, default=16)
    parser.add_argument('--NUM_EPOCHS', type=int, default=2000)
    parser.add_argument('--NUM_WORKERS', type=int, default=16)
    parser.add_argument('--LR', type=float, default=1e-3)
    parser.add_argument('--STOPPING_LR', type=float, default=1e-10)
    parser.add_argument('--MOMENTUM', type=float, default=0.9)
    parser.add_argument('--LR_FACTOR', type=float, default=0.2)
    parser.add_argument('--LR_PATIENCE', type=int, default=5)
    parser.add_argument('--SAVE_PER_EPOCH', type=int, default=5)
    parser.add_argument('--LOSS_CHECK_PER', type=int, default=100)
    
    parser.add_argument('--GPU', type=int, default=0)
    parser.add_argument('--SAVE_DIR_PATH', type=str, default='./')
    parser.add_argument('--LOAD_WEIGHT_FROM', type=str, default=None)

    config = parser.parse_args()
    if config.USE_YAML_CONFIG is not None:
        with open('./cssma_config.yaml', 'r') as f:
            _hparam_dict = yaml.load(f, Loader=yaml.FullLoader)
            hparam_dict = _hparam_dict[config.USE_YAML_CONFIG]
        config.__dict__.update(hparam_dict)
    
    run(config)


