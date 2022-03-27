"""
2021/06/10 Jeong Choi
Embedding Sequence Aggregation Modules. 
- CPC module
- Transformer module
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CPCModule(nn.Module):
    # def __init__(self, timestep, batch_size, seq_len):
    def __init__(self, config):

        super(CPCModule, self).__init__()
        # audio encoder output (batch_size, time_steps, emb_dim)
        self.config = config
        self.num_cpc_input_steps = config.NUM_CPC_INPUT_STEPS
        self.num_cpc_predictions = config.NUM_CPC_PREDICTIONS

        self.gru = nn.GRU(self.config.EMB_DIM, self.config.EMB_DIM, num_layers=1, bidirectional=False, batch_first=True)        
        self.Wk  = nn.ModuleList([nn.Linear(self.config.EMB_DIM, self.config.EMB_DIM) for i in range(self.config.NUM_CPC_PREDICTIONS)])
        self.softmax  = nn.Softmax(dim=-1)
        self.lsoftmax = nn.LogSoftmax(dim=-1)
        self.tanh = nn.Tanh()

        def _weights_init(m):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # initialize GRU
        for layer_p in self.gru._all_weights:
            for p in layer_p:
                if 'weight' in p:
                    nn.init.kaiming_normal_(self.gru.__getattr__(p), mode='fan_out', nonlinearity='relu')

        self.apply(_weights_init)

    def init_hidden(self, batch_size):
        if self.config.GPU is None: 
            return torch.zeros(1, batch_size, self.config.EMB_DIM)
        else:
            return torch.zeros(1, batch_size, self.config.EMB_DIM).cuda(self.config.GPU)
        
    
    # predicting the future
    def forward(self, x, hidden): 
        batch = x.size()[0] 
        # x = x.permute(1, 0, 2)
        if self.config.VERBOSE: print('1) input', x.size())
        if self.config.VERBOSE: print('   hidden', hidden.size())
        
        # randomly pick sequence starting point 
        input_start_timestep = torch.randint(x.size(1) - (self.num_cpc_input_steps + self.num_cpc_predictions), size=(1,)).long() # 11 - 4 : 0~7 사이 랜덤 
        output_start_timestep = input_start_timestep + self.num_cpc_input_steps

        # if self.hparams.verbose: print('  -> downsampled to ', self.seq_len//160)
        if self.config.VERBOSE: print('  -> input_start_timestep: ', input_start_timestep)
        if self.config.VERBOSE: print('  -> output_start_timestep: ', output_start_timestep)
        if self.config.VERBOSE: print('2) hidden', hidden.size())
        
        z = x
        if self.config.VERBOSE: print('3) z', z.size())
        # [batch, time_step, emb_dim]

        # 레이블로 사용할 encoded sample들.
        if self.config.GPU is None: 
            encoded_labels = torch.empty((self.num_cpc_predictions, batch, self.config.EMB_DIM)).float() # e.g. size 12*8*512
        else:
            encoded_labels = torch.empty((self.num_cpc_predictions, batch, self.config.EMB_DIM), device=torch.device(self.config.GPU)).float()
        if self.config.VERBOSE: print('encoded_labels', encoded_labels.size())

        # prepare encoded samples from the output of encoder
        # timestep 동안의 encoded samples (label이 될 것들) -> 이들또한 num_timesteps 만큼의 시퀀스

        for i in np.arange(self.num_cpc_predictions):
            if self.config.VERBOSE: print('z[:,output_start_timestep+i,:]', z[:,output_start_timestep+i,:].transpose(0, 1).size())
            encoded_labels[i] = z[:,output_start_timestep + i,:].permute(1, 0, 2).contiguous()  # .view(batch, self.config.EMB_DIM) # z_tk e.g. size 8*512

        # encoded_labels = encoded_labels.permute(1, 0, 2).contiguous()
        if self.config.VERBOSE: print('encoded_labels transposed', encoded_labels.size())
        # GRU : 랜덤한 길이의 시퀀스가 forward로 주어지고 이에 대해 GRU output을 사용하게 됨.
        forward_seq = z[:, input_start_timestep : input_start_timestep + self.num_cpc_input_steps, :] # e.g. size 8*100*512 seq_starting_timestep 전 까지의 encoded embedding seq
        if self.config.VERBOSE: print('5) forward_seq', forward_seq.size())

        output, hidden = self.gru(forward_seq, hidden) # output size e.g. 8*100*256
        if self.config.VERBOSE: print('6) GRU output, hidden', output.size(), hidden.size())
        # (이 아웃풋 중 마지막 타임스텝의 임베딩을 사용해서 z_(t+k)를 예측.)
        
        # last layer output (gru channel : self.hparams.cpc_emb_dim)
        c_t = output[:,-1,:].view(batch, self.config.EMB_DIM).contiguous() # c_t e.g. size 8*256 
        if self.config.VERBOSE: print('7) c_t', c_t.size())

        # prepare slot for prediction (same number as timestep)
        if self.config.GPU is None: 
            pred = torch.empty((self.num_cpc_predictions, batch, self.config.EMB_DIM)).float() # e.g. size 12*8*512 -> randomly 
        else:
            pred = torch.empty((self.num_cpc_predictions, batch, self.config.EMB_DIM), device=torch.device(self.config.GPU)).float()
        if self.config.VERBOSE: print('8) pred', pred.size())
        
        # self.Wk : future prediction linear layer
        # cpc_timesteps 만큼의 prediction을 위한 linear layer들.
        for i in np.arange(0, self.num_cpc_predictions):
            linear = self.Wk[i] # self.num_timesteps number of linear layer
            pred[i] = linear(c_t) # Wk*c_t e.g. size 8*512
        
        return encoded_labels, pred


    def predict(self, x, hidden):
        z = x
        if self.config.VERBOSE: print('3) z', z.size())
        if self.config.VERBOSE: print('4) z transposed ', z.size())
        
        output, hidden = self.gru(z, hidden) # output size : batch, timesteps, emb_dim
        if self.config.VERBOSE: print('5) output ', output.size())
        if self.config.VERBOSE: print('6) hidden ', hidden.size())
        return output, hidden 
