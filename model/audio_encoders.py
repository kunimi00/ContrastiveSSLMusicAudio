"""
Audio encoders

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MelConv5by5Enc(nn.Module):
    """
    input (batch_size, channels, time, freq) : (batch_size, 1, 188, 96) 
    """
    def __init__(self, config):
        super(MelConv5by5Enc, self).__init__()
        self.config = config
        # hidden_input, hidden_output
        self.hidden_size = [
            (1, 128),
            (128, 256),
            (256, 256),
            (256, 512),
            (512, 512)
        ]
        # kernel_size, stride, padding
        self.kernel_info = [
            (5, 1, 2),
            (5, 1, 2),
            (5, 1, 2),
            (5, 1, 2),
            (5, 1, 2)
        ]
        if config.SEGMENT_LENGTH_SEC == 3:
            # 2d maxpool kernel size
            self.maxpool2d = [
                (2, 2),
                (2, 2),
                (2, 2),
                (4, 4),
                (5, 3)
            ]
        elif config.SEGMENT_LENGTH_SEC == 1.5:
            # 2d maxpool kernel size
            self.maxpool2d = [
                (2, 2),
                (2, 2),
                (2, 2),
                (4, 4),
                (2, 3)
            ]
        
        self.fc_in_outs = [(512, config.EMB_DIM)]
        """
        c.f.
        Frame-level 비교 (chord recognition task) 를 위한 maxpooling 
        : (1, 2), (1, 2), (1, 2), (1, 2), (2, 4)
        """

        """
        input (batch_size, channels, frames, freq) : (batch_size, 1, 188, 96)
        """
        self.sequential = []
        for (h_in, h_out), (kernel_size, stride, padding), max_pool in zip(self.hidden_size, self.kernel_info, self.maxpool2d):
            self.sequential.append(
                nn.Sequential(
                    nn.Conv2d(h_in, h_out, kernel_size=kernel_size, stride=stride, padding=padding),
                    nn.BatchNorm2d(h_out),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=max_pool),
                )
            )
        self.sequential = nn.Sequential(*self.sequential)
        self.fc_layer = nn.Linear(self.fc_in_outs[0][0], self.fc_in_outs[0][1])

        self.apply(self._init_weights)

    def forward(self, x):
        if self.config.VERBOSE: print('x:', x.size())
        x = x.unsqueeze(1)      
        if self.config.VERBOSE: print('x unsqueezed:', x.size())  
        x = self.sequential(x)  #  out : torch.Size([16, 512, T * 1, 1]) -> T timesteps
        if self.config.VERBOSE: print('x conved:', x.size())  
        # assert x.size(2) == self.config.NUM_TIMESTEPS

        # if x.size(2) > 1:  # when using more than 1 sample (sequence)
        x = x.squeeze(-1)  # remove last dim
        out_seq = []
        for idx in range(x.size(2)):
            curr_step_x = x[:, :, idx]
            out_seq.append(self.fc_layer(curr_step_x))
        out_seq = torch.stack(out_seq, dim=1)  # torch.Size([B, Timesteps, Emb dim])
        if self.config.VERBOSE: print('out_seq:', out_seq.size())
        
        if self.config.NEG_STRATEGY == 'intra-track' or self.config.MODEL_INFO == 'CPC':
            if self.config.VERBOSE: print('out:', out_seq.size())
            return out_seq

        elif self.config.NEG_STRATEGY == 'inter-track':
            out = torch.mean(out_seq, dim=1, keepdim=True)
            if self.config.VERBOSE: print('out:', out.size())
            return out

    def _init_weights(self, layer) -> None:
        if isinstance(layer, nn.Conv2d):
            nn.init.kaiming_uniform_(layer.weight)
        elif isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)



class MelConv5by5EncMulti(nn.Module):
    def __init__(self, config):
        super(MelConv5by5EncMulti, self).__init__()
        self.config = config
        # hidden_input, hidden_output
        self.hidden_size = [
            (1, 128),
            (128, 256),
            (256, 256),
            (256, 512),
            (512, config.EMB_DIM)
        ]
        # kernel_size, stride, padding
        self.kernel_info = [
            (5, 1, 2),
            (5, 1, 2),
            (5, 1, 2),
            (5, 1, 2),
            (5, 1, 2)
        ]
        # 2d maxpool kernel size
        if config.SEGMENT_LENGTH_SEC == 3:
            # 2d maxpool kernel size
            self.maxpool2d = [
                (2, 2),
                (2, 2),
                (2, 2),
                (4, 4),
                (5, 3)
            ]
        elif config.SEGMENT_LENGTH_SEC == 1.5:
            # 2d maxpool kernel size
            self.maxpool2d = [
                (2, 2),
                (2, 2),
                (2, 2),
                (4, 4),
                (2, 3)
            ]
        
        self.fc_ins = [6144, 6144, 3072, 1536, config.EMB_DIM]
        self.layer_single_segment_width = [94, 47, 23, 5, 1]

        """
        input (batch_size, channels, frames, freq) : (batch_size, 1, 188, 96)

        """
        fc_layers = []
        for fc_in in self.fc_ins:
            fc_layers.append(nn.Linear(fc_in, config.EMB_DIM))

        self.fc_layers = nn.ModuleList(fc_layers)

        convs = []
        for (h_in, h_out), (kernel_size, stride, padding), max_pool in zip(self.hidden_size, self.kernel_info, self.maxpool2d):
            convs.append(
                nn.Sequential(
                    nn.Conv2d(h_in, h_out, kernel_size=kernel_size, stride=stride, padding=padding),
                    nn.BatchNorm2d(h_out),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=max_pool),
                )
            )
        self.convs = nn.ModuleList(convs)

        self.apply(self._init_weights)

    def forward(self, x, concat=False):
        if self.config.VERBOSE: print('input:', x.size())  # (batch_size, 1, 188, 96) : B, C, T, F
        x = x.unsqueeze(1)
        if self.config.VERBOSE: print('unsqueezed:', x.size())
        linear_outs = []
        if self.config.VERBOSE: print('NUM_TIMESTEPS :', self.config.NUM_TIMESTEPS)
        # layer
        for layer_idx in range(len(self.convs)):
            x = self.convs[layer_idx](x)  # out : torch.Size([B, C, Timesteps * curr_level_single_seg_length, F]) 
            if self.config.VERBOSE: print('layer', layer_idx, 'conv output x.size', x.size())
            if self.config.VERBOSE: print('conv ' + str(layer_idx) + ' :', x.size())

            if self.config.NUM_TIMESTEPS > 1:
                # curr_level_single_seg_length = x.size(2) // self.config.NUM_TIMESTEPS
                curr_level_single_seg_length = self.layer_single_segment_width[layer_idx]
                if self.config.VERBOSE: print('x.size()', x.size())
                if self.config.VERBOSE: print('curr_level_single_seg_length :', curr_level_single_seg_length) 
                """
                [B, C, Timesteps * curr_level_single_seg_length, F]
                """
                curr_level_seq = []
                for time_idx in range(self.config.NUM_TIMESTEPS):
                    if self.config.VERBOSE: print('  seq time_idx', time_idx)
                    if self.config.VERBOSE: print('    layer', layer_idx, curr_level_single_seg_length * time_idx, '~', curr_level_single_seg_length * (time_idx+1))
                    curr_level_seg = x[:, :, curr_level_single_seg_length * time_idx:curr_level_single_seg_length * (time_idx+1), :]
                    if self.config.VERBOSE: print('  curr_level_seg', curr_level_seg.size())
                    curr_level_seg_1d, _  = torch.max(curr_level_seg, 2)  # max pooling along time axis
                    if self.config.VERBOSE: print('  curr_level_seg_1d', curr_level_seg_1d.size())
                    curr_level_seg_unroll = curr_level_seg_1d.view(curr_level_seg_1d.size(0), curr_level_seg_1d.size(1) * curr_level_seg_1d.size(2))
                    if self.config.VERBOSE: print('  curr_level_seg_unroll', curr_level_seg_unroll.size())
                    curr_level_seg_linear_out = self.fc_layers[layer_idx](curr_level_seg_unroll)
                    if self.config.VERBOSE: print('  curr_level_seg_linear_out', curr_level_seg_linear_out.size())
                    curr_level_seq.append(curr_level_seg_linear_out)
                """
                3초 단위 오디오에 대한 conv output -> fc layer output 까지 다 구해놓은 뒤에 averaging을 수행.
                """
                curr_level_seq = torch.stack(curr_level_seq, dim=1)
                if self.config.VERBOSE: print('-> curr_level_seq', curr_level_seq.size())

                if self.config.MODEL_INFO == 'Siamese':
                    if self.config.NEG_STRATEGY == 'inter-track':
                        curr_level_out = torch.mean(curr_level_seq, dim=1, keepdim=False)
                        # assert curr_level_seq.size(1) == self.config.NUM_TIMESTEPS
                        linear_outs.append(curr_level_out)

                    elif self.config.NEG_STRATEGY == 'intra-track':
                        linear_outs.append(curr_level_seq)
                elif self.config.MODEL_INFO == 'CPC' or 'Transformer':
                    linear_outs.append(curr_level_seq)
                    

            else:  # when dealing with a single sample
                """
                input (batch_size, channels, frames, freq) : (batch_size, 1, 188, 96)
                """
                x_1d, _  = torch.max(x, 2)
                # print('x_1d:', x_1d.size())
                x_unroll = x_1d.view(x_1d.size(0), x_1d.size(1) * x_1d.size(2))
                # print('x_unroll:', x_unroll.size())
                linear_out = self.fc_layers[layer_idx](x_unroll)
                linear_outs.append(linear_out)

        if concat:
            out_concat = torch.cat(linear_outs, -1)
            # print('out_concat', out_concat.size())
            return out_concat
        else:
            return linear_outs

    def _init_weights(self, layer) -> None:
        if isinstance(layer, nn.Conv2d):
            nn.init.kaiming_uniform_(layer.weight)
        elif isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)



"""
Reference : SampleCNN used in CLMR (https://github.com/Spijkervet/CLMR)
"""

class SampleCNN(nn.Module):
    def __init__(self, strides, supervised, out_dim):
        super(SampleCNN, self).__init__()

        self.strides = strides
        self.supervised = supervised
        self.sequential = [
            nn.Sequential(
                nn.Conv1d(1, 128, kernel_size=3, stride=3, padding=0),
                nn.BatchNorm1d(128),
                nn.ReLU(),
            )
        ]

        self.hidden = [
            [128, 128],
            [128, 128],
            [128, 256],
            [256, 256],
            [256, 256],
            [256, 256],
            [256, 256],
            [256, 256],
            [256, 512],
        ]

        assert len(self.hidden) == len(
            self.strides
        ), "Number of hidden layers and strides are not equal"
        for stride, (h_in, h_out) in zip(self.strides, self.hidden):
            self.sequential.append(
                nn.Sequential(
                    nn.Conv1d(h_in, h_out, kernel_size=stride, stride=1, padding=1),
                    nn.BatchNorm1d(h_out),
                    nn.ReLU(),
                    nn.MaxPool1d(stride, stride=stride),
                )
            )

        # 1 x 512
        self.sequential.append(
            nn.Sequential(
                nn.Conv1d(512, 512, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm1d(512),
                nn.ReLU(),
            )
        )

        self.sequential = nn.Sequential(*self.sequential)

        if self.supervised:
            self.sigmoid = nn.Sigmoid()
            self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(512, out_dim)

    def forward(self, x):
        out = self.sequential(x)
        if self.supervised:
            out = self.dropout(out)

        out = out.reshape(x.shape[0], out.size(1) * out.size(2))
        logit = self.fc(out)

        if self.supervised:
            logit = self.sigmoid(logit)

        return logit




