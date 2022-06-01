"""
Contrastive algorithms
- Siamese network
- CPC network
"""

import sys
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from model.audio_encoders import *
from model.emb_aggregators import *
from model.abstract_model import AbstractModel
from util.util_data import *

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchsummary import summary

import numpy as np
from tqdm import tqdm
from glob import glob
import random

class ModelSiamese(AbstractModel):

    def __init__(self, config, audio_encoder):
        AbstractModel.__init__(self)
        self.config = config
        self.encoder = audio_encoder
        
        self.softmax  = nn.Softmax(dim=-1)
        self.lsoftmax = nn.LogSoftmax(dim=-1)
        self.tanh = nn.Tanh()

        self.num_epochs = self.config.NUM_EPOCHS
        self.criterion = nn.CrossEntropyLoss()  

        self.encoder.cuda(self.config.GPU)
        self.criterion.cuda(self.config.GPU)

        params = list(self.encoder.parameters())
        
        self.curr_learning_rate = self.config.LR
        self.optimizer = optim.Adam(params, lr=self.curr_learning_rate)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=self.config.LR_FACTOR, patience=self.config.LR_PATIENCE, verbose=True)
        

    def preprocess(self):
        """
        Preprocess 
        """
        pass

 
    def early_stop(self, loss, epoch):
        self.scheduler.step(loss, epoch)
        self.curr_learning_rate = self.optimizer.param_groups[0]['lr']
        stop = self.curr_learning_rate < self.config.STOPPING_LR

        return stop

    def _dot_product(self, a, b):
        print(a.unsqueeze(1).size(), a.unsqueeze(1).device)
        print(b.unsqueeze(2).size(), b.unsqueeze(2).device)
        return torch.matmul(a.unsqueeze(1), b.unsqueeze(2))

    def train(self, dataloader, epoch_from=0):
        
        self.encoder.train()

        _check = self.config.LOSS_CHECK_PER
        for epoch in range(self.num_epochs + epoch_from):
            epoch += epoch_from
            print('epoch', epoch)
            
            epoch_loss = 0
            running_loss, running_acc = 0, 0
            
            for batch_idx, (x, tr_ids) in tqdm(enumerate(dataloader)):
                anchor = x[0]
                pos = x[1]
                batch = anchor.size()[0] 

                # (1) Audio encoding
                if self.config.GPU is not None:
                    anchor = anchor.to(self.config.GPU)
                    pos = pos.to(self.config.GPU)
                
                if self.config.AUDIO_ENCODER == 'MelConv5by5Enc':
                    anchor = self.encoder(anchor)  # out : [B, Timesteps, Emb dim]
                    pos = self.encoder(pos)
                    if self.config.VERBOSE: print('encoded anchor', anchor.size())
                    if self.config.VERBOSE: print('encoded pos', pos.size())
                
                elif self.config.AUDIO_ENCODER == 'MelConv5by5EncMulti':
                    anchor_list = self.encoder(anchor, concat=False)
                    pos_list = self.encoder(pos, concat=False)
                    if self.config.VERBOSE: print('encoded anchor list (single element)', anchor_list[0].size())
                    if self.config.VERBOSE: print('encoded pos list (single element)', pos_list[0].size())

                
                # (2) Siamese loss 
                # (2-1) inter-track loss
                if self.config.NEG_STRATEGY == 'inter-track':
                    if self.config.AUDIO_ENCODER == 'MelConv5by5Enc':
                        # [B, Timesteps, Emb dim]
                        nce, accuracy = 0, 0
                        _num_timesteps = anchor.size(1)
                        curr_anchor, curr_pos = anchor.squeeze(1), pos.squeeze(1)
                        total = torch.mm(curr_anchor, torch.transpose(curr_pos,0,1)) # e.g. size 8*8
                        if self.config.VERBOSE: print('total matrix', total.size())
                        correct = torch.sum(torch.eq(torch.argmax(self.softmax(total), dim=0), torch.arange(0, batch).to(self.config.GPU))) # correct is a tensor
                        curr_nce = torch.sum(torch.diag(self.lsoftmax(total))) # nce is a tensor
                        curr_nce /= -1.*batch
                        
                        loss = curr_nce
                        accuracy = 1.* correct.item()/batch
                        
                    elif self.config.AUDIO_ENCODER == 'MelConv5by5EncMulti':
                        loss = 0
                        accuracy = 0
                        for _idx in range(len(anchor_list)):
                            anchor, pos = anchor_list[_idx], pos_list[_idx]
                            if self.config.VERBOSE: print('anchor-'+str(_idx)+':', anchor.size())
                            curr_anchor, curr_pos = anchor, pos
                            total = torch.mm(curr_anchor, torch.transpose(curr_pos,0,1)) # e.g. size 8*8
                            if self.config.VERBOSE: print('total matrix', total.size())
                            correct = torch.sum(torch.eq(torch.argmax(self.softmax(total), dim=0), torch.arange(0, batch).to(self.config.GPU))) # correct is a tensor
                            curr_nce = torch.sum(torch.diag(self.lsoftmax(total))) # nce is a tensor
                            curr_nce /= -1.*batch
                            
                            loss += curr_nce
                            accuracy += 1.* correct.item()/batch

                        loss = loss / len(anchor_list)
                        accuracy = accuracy / len(anchor_list)


                # (2-2) intra-track loss
                elif self.config.NEG_STRATEGY == 'intra-track':
                    if self.config.AUDIO_ENCODER == 'MelConv5by5Enc':
                        nce, accuracy = 0, 0
                        _num_timesteps = anchor.size(1)

                        # curr_pos_position = torch.randint(low=0, high=_num_timesteps, size=(1,)).to(self.config.GPU)
                        curr_pos_position = random.randint(0, _num_timesteps-1)
                        # [B, Timesteps, Emb dim]
                        anchor_x = anchor[:, curr_pos_position, :].to(self.config.GPU)
                        pos_x = pos[:, curr_pos_position, :].to(self.config.GPU)
                        if self.config.VERBOSE: print('anchor_x', anchor_x.size(), anchor_x.device)
                        if self.config.VERBOSE: print('pos_x', pos_x.size(), pos_x.device)
                        l_pos = torch.bmm(anchor_x.view(-1, 1, self.config.EMB_DIM), pos_x.view(-1, self.config.EMB_DIM, 1)) / self.config.INFONCE_TAO
                        l_pos = l_pos.squeeze()
                        # l_pos = (anchor_x*pos_x).sum(-1)
                        if self.config.VERBOSE: print('l_pos', l_pos.size())
                        
                        l_negs = []
                        for _position in range(_num_timesteps):
                            if curr_pos_position == _position:
                                continue
                            neg_x = anchor[:, _position, :].to(self.config.GPU)
                            if self.config.VERBOSE: print('neg_x', neg_x.size())
                            l_neg = torch.bmm(anchor_x.view(-1, 1, self.config.EMB_DIM), neg_x.view(-1, self.config.EMB_DIM, 1)) / self.config.INFONCE_TAO
                            l_neg = l_neg.squeeze()
                            # l_neg = (anchor_x*neg_x).sum(-1)
                            if self.config.VERBOSE: print('l_neg', l_neg.size())
                            l_negs.append(l_neg)

                        logits = torch.stack([l_pos] + l_negs, dim=-1)
                        if self.config.VERBOSE: print('logits', logits.size())
                        labels = torch.zeros(batch, dtype=torch.long).to(self.config.GPU)
                        if self.config.VERBOSE: print('labels', labels.size())
                        correct = torch.sum(torch.eq(torch.argmax(self.softmax(logits), dim=1), labels))
                        loss = self.criterion(logits, labels)
                        accuracy = 1.* correct.item()/batch
                    

                    elif self.config.AUDIO_ENCODER == 'MelConv5by5EncMulti':
                        loss = 0
                        accuracy = 0

                        for _idx in range(len(anchor_list)):  # per level 
                            anchor, pos = anchor_list[_idx], pos_list[_idx]
                            if self.config.VERBOSE: print('anchor-'+str(_idx)+':', anchor.size())
                            _num_timesteps = anchor.size(1)

                            # for pos_idx in range(_num_timesteps):
                            pos_idx = random.randint(0, _num_timesteps-1)

                            anchor_x, pos_x = anchor[:, pos_idx, :], pos[:, pos_idx, :]  # curr pos timestep
                            l_pos = torch.bmm(anchor_x.view(-1, 1, self.config.EMB_DIM), pos_x.view(-1, self.config.EMB_DIM, 1)) / self.config.INFONCE_TAO
                            l_pos = l_pos.squeeze()
                            
                            l_negs = []
                            for neg_idx in range(_num_timesteps):
                                if neg_idx == pos_idx:
                                    continue
                                neg_x = anchor[:, neg_idx, :].to(self.config.GPU)
                                if self.config.VERBOSE: print('neg_x', neg_x.size())
                                l_neg = torch.bmm(anchor_x.view(-1, 1, self.config.EMB_DIM), neg_x.view(-1, self.config.EMB_DIM, 1)) / self.config.INFONCE_TAO
                                l_neg = l_neg.squeeze()
                                # l_neg = (anchor_x*neg_x).sum(-1)
                                if self.config.VERBOSE: print('l_neg', l_neg.size())
                                l_negs.append(l_neg)
                            
                            logits = torch.stack([l_pos] + l_negs, dim=-1)
                            if self.config.VERBOSE: print('logits', logits.size())
                            labels = torch.zeros(batch, dtype=torch.long).to(self.config.GPU)
                            if self.config.VERBOSE: print('labels', labels.size())
                            correct = torch.sum(torch.eq(torch.argmax(self.softmax(logits), dim=1), labels))
                            curr_loss = self.criterion(logits, labels)
                            curr_accuracy = 1.* correct.item()/batch

                            loss += curr_loss
                            accuracy += curr_accuracy

                        loss = loss / len(anchor_list)
                        accuracy = accuracy / len(anchor_list)
                        
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                running_loss += loss.item()
                running_acc += accuracy

                epoch_loss += self.config.BATCH_SIZE * loss.item()
                if batch_idx % _check == 0 and batch_idx != 0:
                    print('[Epoch %d, Batch %5d] loss : %.6f, acc : %.3f' % \
                        (epoch, batch_idx, 
                        running_loss/_check,
                        running_acc/_check))
                    
                    running_loss, running_acc = 0, 0
            
            epoch_loss = epoch_loss / len(dataloader.dataset)
            print('EXP :', self.config.CURR_WEIGHT_FOLDER)
            print("[Epoch %04d/%d] [Train Loss: %.4f]" % (epoch, self.config.NUM_EPOCHS + epoch_from, epoch_loss))
            print("curr learning rate :", self.optimizer.param_groups[0]['lr'])
            
            if epoch % self.config.SAVE_PER_EPOCH == 0 and epoch != 0:
                self.save_model(epoch, epoch_loss)

            if self.early_stop(epoch_loss, epoch + 1):
                print('Training done (early stop).')        
                break
        
        print('Training done.')
 
    def predict(self, dataloader):
        
        """
        Predict
        """
        
        mix_id_to_emb_dict = {}
        print('Retrieve', len(dataloader.dataset), 'mix embeddings /', len(dataloader), 'batches.')

        weight_folder = self.config.LOAD_WEIGHT_FROM.split('/')[-2]
        weight_file = self.config.LOAD_WEIGHT_FROM.split('/')[-1]
        curr_epoch = int(weight_file.split('_')[2])

        save_folder = os.path.join(self.config.INFERENCE_PATH, self.config.INFERENCE_DATA, weight_folder)
        save_path = os.path.join(save_folder, 'tr_id_to_emb_dict_ep' + str(curr_epoch) + '.p')
        print(save_path)

        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
            print('mkdir done.')
        
        emb_path = os.path.join(save_folder, 'embeds')
        if not os.path.exists(emb_path):
            os.mkdir(emb_path)
            print('mkdir done.') 
        
        self.encoder.eval()
        
        for batch_idx, (x, tr_ids) in tqdm(enumerate(dataloader)):
            anchor = x
            if self.config.VERBOSE: print(anchor.size())
            if self.config.GPU is not None:
                anchor = anchor.to(self.config.GPU)

            if self.config.MEL_ENCODER == 'MelConv5by5Enc':
                encoded = self.encoder(anchor)  # out : [B, Timesteps, Emb dim]
                # encoded = encoded.permute(0,2,1)
                if self.config.VERBOSE: print('encoded', encoded.size())

            elif self.config.MEL_ENCODER == 'MelConv5by5EncMulti':
                encoded = self.encoder(anchor, concat=True)  # out : [B, Timesteps, Emb dim * num_layers(5)]
                # encoded = torch.stack(encoded_list, dim=1)
                if self.config.VERBOSE: print('encoded', encoded.size())
            
            for idx in range(len(tr_ids)):
                curr_mid = tr_ids[idx]
                emb = encoded[idx].detach().cpu().clone().numpy()
                if self.config.VERBOSE: print('inference mid:', curr_mid, emb.shape)
                mix_id_to_emb_dict[curr_mid] = emb
        
        save_obj_to_pickle(mix_id_to_emb_dict, save_path)
        print('Done saving : ', save_path)
        
        
    def load_model(self, weight_path):
        print(weight_path)
        self.encoder.load_state_dict(torch.load(weight_path, map_location=self.config.DEVICE))
        
        print('Loading model done.')


    def save_model(self, epoch, epoch_loss):
        """
        Save the model
        """
        curr_weight_name_encoder =  os.path.join(self.config.WEIGHT_PATH, self.config.CURR_WEIGHT_FOLDER, 'w_ep-%05d_l-%.6f' % (epoch, epoch_loss))
        if not os.path.exists(os.path.dirname(curr_weight_name_encoder)):
            os.makedirs(os.path.dirname(curr_weight_name_encoder))

        torch.save(self.encoder.state_dict(), curr_weight_name_encoder + '.pth')
 
    def save_result(self, result):
        """
        Save the inference result
        """
        pass
 
    def postprocess(self):
        """
        Postprocess
        """
        pass




class ModelCPC(AbstractModel):

    def __init__(self, config, audio_encoder, cpc_module):
        AbstractModel.__init__(self)
        self.config = config
        self.encoder = audio_encoder
        if self.config.AUDIO_ENCODER == 'MelConv5by5Enc':
            self.cpc_module = cpc_module
            self.cpc_module.cuda(self.config.GPU)
        elif self.config.AUDIO_ENCODER == 'MelConv5by5EncMulti':
            self.cpc_module_list = cpc_module
            for idx in range(len(self.cpc_module_list)):
                self.cpc_module_list[idx].cuda(self.config.GPU)
        
        self.num_epochs = self.config.NUM_EPOCHS
        self.criterion = nn.CrossEntropyLoss()  # Masked LM 용

        self.encoder.cuda(self.config.GPU)
        self.criterion.cuda(self.config.GPU)

        self.softmax  = nn.Softmax(dim=-1)
        self.lsoftmax = nn.LogSoftmax(dim=-1)
        if self.config.AUDIO_ENCODER == 'MelConv5by5Enc':
            params = list(self.encoder.parameters()) + list(self.cpc_module.parameters())
        elif self.config.AUDIO_ENCODER == 'MelConv5by5EncMulti':
            params = [] 
            params.extend(list(self.encoder.parameters())) 
            for idx in range(len(self.cpc_module_list)):
                params.extend(list(self.cpc_module_list[idx].parameters()))
        
        self.curr_learning_rate = self.config.LR
        self.optimizer = optim.Adam(params, lr=self.curr_learning_rate)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=self.config.LR_FACTOR, patience=self.config.LR_PATIENCE, verbose=True)


    def preprocess(self):
        """
        Preprocess 
        """
        pass

 
    def early_stop(self, loss, epoch):
        self.scheduler.step(loss, epoch)
        self.curr_learning_rate = self.optimizer.param_groups[0]['lr']
        stop = self.curr_learning_rate < self.config.STOPPING_LR

        return stop


    def train(self, dataloader, epoch_from=0):
        """
        Train
        """
        self.encoder.train()
        if self.config.AUDIO_ENCODER == 'MelConv5by5Enc':
            self.cpc_module.train()
        elif self.config.AUDIO_ENCODER == 'MelConv5by5EncMulti':
            for idx in range(len(self.cpc_module_list)):
                self.cpc_module_list[idx].train()
        
        _check = self.config.LOSS_CHECK_PER

        for epoch in range(self.num_epochs + epoch_from):
            epoch += epoch_from
            print('epoch', epoch)
            
            epoch_loss = 0
            running_loss, running_acc = 0, 0
            
            for batch_idx, (x, tr_ids) in tqdm(enumerate(dataloader)):
                anchor = x
                if self.config.VERBOSE: print('anchor', anchor.size())
                batch = anchor.size()[0] 
                
                # (1) Audio encoding
                if self.config.GPU is not None:
                    anchor = anchor.to(self.config.GPU)
                
                if self.config.AUDIO_ENCODER == 'MelConv5by5Enc':
                    anchor = self.encoder(anchor)  # out : [B, Timesteps, Emb dim]
                    if self.config.VERBOSE: print('encoded anchor', anchor.size())
                
                elif self.config.AUDIO_ENCODER == 'MelConv5by5EncMulti':
                    anchor_list = self.encoder(anchor, concat=False)
                    if self.config.VERBOSE: print('encoded anchor list - single element size : ', anchor_list[0].size())
                    # ([8, 11, 128])


                # (2) CPC aggregation
                if self.config.AUDIO_ENCODER == 'MelConv5by5Enc':
                    hidden = self.cpc_module.init_hidden(len(anchor)).to(self.config.GPU)
                    encoded_labels, pred = self.cpc_module(anchor, hidden)
                    _tmp_label = torch.arange(start=0, end=batch, device=torch.device(self.config.GPU))
                    nce_all, acc_all = 0, 0
                    for i in np.arange(0, self.config.NUM_CPC_PREDICTIONS):
                        # matrix multiplication으로 encoded sample들(레이블)과 pred과의 dot product의 결과 값. (이것이 가장 높은 것을 pick)
                        total = torch.mm(encoded_labels[i], torch.transpose(pred[i],0,1)) # e.g. size 8*8
                        if self.config.VERBOSE: print('-> total', i, ':', total.size(), total.device)
                        # softmax의 argmax로 선택. -> element-wise equality (True, False) -> torch sum으로 num correct를 얻음.
                        correct = torch.sum(torch.eq(torch.argmax(self.softmax(total), dim=0), _tmp_label))
                        # loss 의 경우 log softmax (axis=0을 따라) -> 이후 diagonal element 만을 sum. (이것이 true 값이기 때문)
                        nce = torch.sum(torch.diag(self.lsoftmax(total))) 
                        # 즉, negative sample 들은 배치 내에서만 부여됨. (16개중 1개를 선택)
                    
                        nce /= -1.*batch
                        acc = 1.* correct.item()/batch
                        nce_all += nce
                        acc_all += acc

                    loss = nce_all / self.config.NUM_CPC_PREDICTIONS
                    accuracy = acc_all / self.config.NUM_CPC_PREDICTIONS

                elif self.config.AUDIO_ENCODER == 'MelConv5by5EncMulti':
                    loss, accuracy = 0, 0
                    for _idx in range(len(anchor_list)):  # per level 
                        anchor = anchor_list[_idx]
                        # curr_l_cpc_module = self.cpc_module_list[_idx]
                        # print('curr_l_cpc_module', curr_l_cpc_module)
                        hidden = self.cpc_module_list[_idx].init_hidden(len(anchor)).to(self.config.GPU)
                        # print('anchor', anchor.size())
                        # print('hidden', hidden.size())
                        encoded_labels, pred = self.cpc_module_list[_idx](anchor, hidden)
                        _tmp_label = torch.arange(start=0, end=batch, device=torch.device(self.config.GPU))

                        nce_all, acc_all = 0, 0
                        for i in np.arange(0, self.config.NUM_CPC_PREDICTIONS):
                            total = torch.mm(encoded_labels[i], torch.transpose(pred[i],0,1)) # e.g. size 8*8
                            if self.config.VERBOSE: print('-> total', i, ':', total.size(), total.device)
                            correct = torch.sum(torch.eq(torch.argmax(self.softmax(total), dim=0), _tmp_label))
                            nce = torch.sum(torch.diag(self.lsoftmax(total))) 
                            nce /= -1.*batch
                            acc = 1.* correct.item()/batch
                            nce_all += nce
                            acc_all += acc

                        loss += nce_all / self.config.NUM_CPC_PREDICTIONS
                        accuracy += acc_all / self.config.NUM_CPC_PREDICTIONS
                    loss /= len(anchor_list)
                    accuracy /= len(anchor_list)

                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                running_loss += loss.item()
                running_acc += accuracy

                epoch_loss += self.config.BATCH_SIZE * loss.item()
                if batch_idx % _check == 0 and batch_idx != 0:

                    # if self.config.TRANSFORMER_NEG_STRATEGY == 'both':
                    print('[Epoch %d, Batch %5d] loss : %.6f, acc : %.3f' % \
                        (epoch, batch_idx, 
                        running_loss/_check,
                        running_acc/_check))
                    
                    running_loss, running_acc = 0, 0
            
            epoch_loss = epoch_loss / len(dataloader.dataset)
            print('EXP :', self.config.CURR_WEIGHT_FOLDER)
            print("[Epoch %04d/%d] [Train Loss: %.4f]" % (epoch, self.config.NUM_EPOCHS + epoch_from, epoch_loss))
            print("curr learning rate :", self.optimizer.param_groups[0]['lr'])
            
            if epoch % self.config.SAVE_PER_EPOCH == 0 and epoch != 0:
                self.save_model(epoch, epoch_loss)

            if self.early_stop(epoch_loss, epoch + 1):
                print('Training done (early stop).')        
                break

        print('Training done.')

 
    def predict(self, dataloader):        
        """
        Predict
        """
        mix_id_to_emb_dict = {}
        print('Retrieve', len(dataloader.dataset), 'mix embeddings /', len(dataloader), 'batches.')
        self.encoder.eval()
        if self.config.AUDIO_ENCODER == 'MelConv5by5Enc':
            self.cpc_module.eval()
        elif self.config.AUDIO_ENCODER == 'MelConv5by5EncMulti':
            for idx in range(len(self.cpc_module_list)):
                self.cpc_module_list[idx].eval()
        
        for batch_idx, (x, tr_ids) in tqdm(enumerate(dataloader)):
            anchor = x
            if self.config.VERBOSE: print('anchor', anchor.size())
            
            # (1) Audio encoding
            if self.config.GPU is not None:
                anchor = anchor.to(self.config.GPU)
            
            if self.config.AUDIO_ENCODER == 'MelConv5by5Enc':
                anchor = self.encoder(anchor)  # out : [B, Timesteps, Emb dim]
                hidden = self.cpc_module.init_hidden(len(anchor)).to(self.config.GPU)
                out_emb_seq, _ = self.cpc_module.predict(anchor, hidden)
                for idx in range(len(tr_ids)):
                    curr_mid = int(tr_ids[idx].detach().cpu().clone().numpy())
                    emb = out_emb_seq[idx].detach().cpu().clone().numpy()
                    # print('inference mid:', curr_mid, emb.shape)
                    mix_id_to_emb_dict[curr_mid] = emb.tolist()
            
            elif self.config.AUDIO_ENCODER == 'MelConv5by5EncMulti':
                anchor_list = self.encoder(anchor, concat=False)
                hidden = self.cpc_module_list[0].init_hidden(len(anchor_list[0])).to(self.config.GPU)

                # initialize list in dictionary 
                for b_idx in range(len(tr_ids)):
                    curr_mid = int(tr_ids[idx].detach().cpu().clone().numpy())
                    mix_id_to_emb_dict[curr_mid] = []

                for l_idx in range(len(anchor_list)):  # per level 
                    anchor = anchor_list[l_idx]
                    curr_l_cpc_module = self.cpc_module_list[l_idx]
                    out_emb_seq, _ = curr_l_cpc_module.predict(anchor, hidden)
                    for b_idx in range(len(tr_ids)):
                        curr_mid = int(tr_ids[b_idx].detach().cpu().clone().numpy())
                        emb = out_emb_seq[b_idx].detach().cpu().clone().numpy()
                        mix_id_to_emb_dict[curr_mid].append(emb.tolist())     
            
        print('done inference.')
        return mix_id_to_emb_dict
 

    def load_model(self, epoch=0):
        if epoch == 0:
            print('Epoch should be specified.')

        else:
            curr_weight_name_encoder = glob(os.path.join(self.config.WEIGHT_PATH, self.config.CURR_WEIGHT_FOLDER, 'encoder_w_ep-%05d_l-*' % (epoch)))[0]
            print('Loading encoder weights from..')
            print(curr_weight_name_encoder)
            self.encoder.load_state_dict(torch.load(curr_weight_name_encoder, map_location=self.config.DEVICE))
            
            if self.config.AUDIO_ENCODER == 'MelConv5by5Enc':
                curr_weight_name_cpc_module = glob(os.path.join(self.config.WEIGHT_PATH, self.config.CURR_WEIGHT_FOLDER, 'cpc-module_w_ep-%05d_l-*' % (epoch)))[0]
                print('Loading cpc module weights from..')
                print(curr_weight_name_cpc_module)
                self.cpc_module.load_state_dict(torch.load(curr_weight_name_cpc_module, map_location=self.config.DEVICE))

            elif self.config.AUDIO_ENCODER == 'MelConv5by5EncMulti':
                for idx in range(len(self.cpc_module_list)):
                    curr_weight_name_cpc_module = glob(os.path.join(self.config.WEIGHT_PATH, self.config.CURR_WEIGHT_FOLDER, 'cpc-module-' + str(idx) + '_w_ep-%05d_l-*' % (epoch)))[0]
                    self.cpc_module_list[idx].load_state_dict(torch.load(curr_weight_name_cpc_module, map_location=self.config.DEVICE))
                    print('cpc module ' + str(idx) +  ' loaded from :', curr_weight_name_cpc_module)

            print('Loading model done.')


    def save_model(self, epoch, epoch_loss):
        """
        Save the model
        """
        curr_weight_name_encoder =  os.path.join(self.config.WEIGHT_PATH, self.config.CURR_WEIGHT_FOLDER, 'encoder_w_ep-%05d_l-%.6f' % (epoch, epoch_loss))
        if not os.path.exists(os.path.dirname(curr_weight_name_encoder)):
            os.makedirs(os.path.dirname(curr_weight_name_encoder))
        
        torch.save(self.encoder.state_dict(), curr_weight_name_encoder + '.pth')
        print('audio encoder saved at :', curr_weight_name_encoder)

        if self.config.AUDIO_ENCODER == 'MelConv5by5Enc':
            curr_weight_name_cpc_module = os.path.join(self.config.WEIGHT_PATH, self.config.CURR_WEIGHT_FOLDER, 'cpc-module_w_ep-%05d_l-%.6f' % (epoch, epoch_loss))
            torch.save(self.cpc_module.state_dict(), curr_weight_name_cpc_module + '.pth')
            print('cpc module saved at :', curr_weight_name_cpc_module)
        elif self.config.AUDIO_ENCODER == 'MelConv5by5EncMulti':
            for idx in range(len(self.cpc_module_list)):
                curr_weight_name_cpc_module = os.path.join(self.config.WEIGHT_PATH, self.config.CURR_WEIGHT_FOLDER, 'cpc-module-' + str(idx) + '_w_ep-%05d_l-%.6f' % (epoch, epoch_loss))
                torch.save(self.cpc_module_list[idx].state_dict(), curr_weight_name_cpc_module + '.pth')
                print('cpc module ' + str(idx) +  ' saved at :', curr_weight_name_cpc_module)



 
    def save_result(self, result):
        """
        Save the inference result
        """
        pass
 
    def postprocess(self):
        """
        Postprocess
        """
        pass






