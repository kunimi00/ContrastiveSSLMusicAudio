"""
2021/06/01 Jeong Choi

Weight, inferenced embedding 을 위한 Path 생성 및 parsing 관련 함수 

"""
def parse_weight_path(_path):
    print('Loading from :', _path)
    weight_folder = _path.split('/')[-2]
    weight_file = _path.split('/')[-1]  # 'w_ep-%05d_l-%.6f' % (epoch, epoch_loss)
    epoch_from = int(weight_file.split('_')[1][3:])
    
    exp_info = weight_folder.split('_')[0][4:]
    model_info = weight_folder.split('_')[1]
    input_repr = weight_folder.split('_')[2]
    audio_encoder = weight_folder.split('_')[3]
    loss_fn = weight_folder.split('_')[4]
    pos_strategy = weight_folder.split('_')[5]
    neg_strategy = weight_folder.split('_')[6]
    if weight_folder.split('_')[7] == 'pos-wav-aug':
        wav_aug = True
    elif weight_folder.split('_')[7] == 'pos-no-aug':
        wav_aug = False
    emb_dim = weight_folder.split('_')[8][4:]
    return exp_info, model_info, input_repr, audio_encoder, loss_fn, pos_strategy, neg_strategy, wav_aug, emb_dim, weight_file, epoch_from


def create_weight_folder_name(config):
    if config.POS_WAV_AUG:
        pos_wav_aug = 'wav-aug'
    else:
        pos_wav_aug = 'no-aug'
    
    return 'exp-{}_{}_{}_{}_{}_{}_{}_pos-{}_emb-{}'.format(config.EXP_INFO,
                                             config.MODEL_INFO,
                                             config.AUDIO_REPR,
                                             config.AUDIO_ENCODER,
                                             config.LOSS_FN,
                                             config.POS_STRATEGY,
                                             config.NEG_STRATEGY,
                                             pos_wav_aug,
                                             config.EMB_DIM)
