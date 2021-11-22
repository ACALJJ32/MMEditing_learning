import torch
from mmedit.models.backbones.sr_backbones import EncoderDecoderNet, Decoder
from copy import deepcopy

model_path = '/media/test/8026ac84-a5ee-466b-affa-f8c81a423d9b/ljj/VSR/mmediting_cuc/work_dirs/encoder_decoder_x4_600k_reds/step1/iter_60000.pth'

load_net = torch.load(model_path, map_location=lambda storage, loc: storage)

load_net = load_net['state_dict']

choose_key = 'decoder'

for key, value in deepcopy(load_net).items():

    key_list = key.split('.')
    if choose_key in key_list:
        tmp_key = key[18:]
        load_net[tmp_key] = value
        
    load_net.pop(key)

model = Decoder(mid_channels=64)

model.load_state_dict(load_net, strict=True)

model_save_path = 'decoder_iter_600000.pth'

torch.save(model.state_dict(), model_save_path)