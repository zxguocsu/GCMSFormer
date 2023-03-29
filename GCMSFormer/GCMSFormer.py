# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 21:04:41 2022

@author: fanyingjie
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import matplotlib.pyplot as plt
import seaborn
import os
seaborn.set_context(context="talk")
from GCMSFormer.da import data_gen, subsequent_mask
import pickle
import matplotlib.ticker as mtick

class Embeddings(nn.Module):
    def __init__(self, d_model):
        super(Embeddings, self).__init__()
        self.d_model = d_model

    def forward(self, x):
        return x * math.sqrt(self.d_model) 

class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], 
                         requires_grad=False)
        return self.dropout(x)

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class Encoder(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)

class Decoder(nn.Module):
    "Generic N layer decoder with masking."
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)

class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)
 
    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)

def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k 
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        
        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn = attention(query, key, value, mask=mask, 
                                 dropout=self.dropout)
        
        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        del query
        del key
        del value
        return self.linears[-1](x)

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many 
    other models.
    """
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator
        
    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(src, src_mask), src_mask,
                            tgt, tgt_mask)
    
    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)
    
    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)
    
class Generator(nn.Module):
    "Define standard linear + softmax generation step. /nn.CrossEntropyLoss"
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return self.proj(x)

def make_model(tgt_vocab, N=4, 
               d_model=1000, d_ff=1024, h=8, dropout=0.1):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), 
                             c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model), c(position)),
        nn.Sequential(Embeddings(d_model), c(position)),
        Generator(d_model, tgt_vocab))
    
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
    return model

class LabelSmoothing(nn.Module):
    "Implement label smoothing."
    def __init__(self, size, padding_idx, smoothing=0.1):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False)
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None
        
    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))
    
class SimpleLossCompute:
    "A simple loss compute and train function. /nn.CrossEntropyLoss"
    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt
        
    def __call__(self, x, y):
        x = self.generator(x)
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)),
                              y.contiguous().view(-1))
        if self.opt is not None:
            loss.backward()
            self.opt.step()
            self.opt.zero_grad()
        return loss.item()

class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        
    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        
    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))
        
def get_std_opt(model):
    return NoamOpt(model.src_embed[0].d_model, 2, 4000,
            torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

def train(data_iter, batch_size, model, loss_compute, epoch, train_src):
    model.train()
    total_loss = 0
    total_losss = 0
    start = time.time()
    for i, batch in enumerate(data_iter):
        out = model.forward(batch.src, batch.trg, 
                            batch.src_mask, batch.trg_mask)
        loss = loss_compute(out, batch.trg_y_ind)
        total_loss += loss
        total_losss += loss
        log_interval = 20#50
        if i % log_interval == 0 and i > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start
            print('| epoch {:3d} | {:5d}/{:5d} batches | '
                  's/batch {:5.2f} | '
                  'loss {:5.2f} | ppl {:8.2f}'.format(
                      epoch, i , len(train_src) // batch_size, elapsed / log_interval,
                      cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start = time.time()
    return total_losss / i

def evaluate(data_iter, eval_model, loss_compute):
    eval_model.eval()
    total_loss = 0
    with torch.no_grad():
        for i, batch in enumerate(data_iter):
            out = eval_model.forward(batch.src, batch.trg, 
                                batch.src_mask, batch.trg_mask)
            loss = loss_compute(out, batch.trg_y_ind)
            total_loss += loss
            del loss
    return total_loss / i

def train_model(para,TRAIN, VALID,tgt_vacob):
    # load Datasets
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tgt_vacob = tgt_vacob.to(device)
    d_models = int(max(para['mz_range']))
    train_src, train_tgt, train_tgt_ind, train_total = TRAIN
    valid_src, valid_tgt, valid_tgt_ind, valid_total = VALID
    best_val_loss = float("inf")
    best_model = None
    # Initialization model
    V = len(tgt_vacob)
    model = make_model(V, N=para['layer_num'], h=para['head']).to(device)
    # loss function
    criterion = nn.CrossEntropyLoss()
    # optimizer and lr
    model_opt = torch.optim.AdamW(model.parameters(), lr=para['lr'])
    scheduler = torch.optim.lr_scheduler.StepLR(model_opt, 1.0, gamma=0.95)
    # train and evaluate model
    t_loss = []
    v_loss = []
    for epoch in range(1, para['epoch']+1):
        epoch_start_time = time.time()
        train_loss = train(data_gen(train_src, train_tgt, para['batch'], train_tgt_ind, d_models, device), para['batch'], model, 
              SimpleLossCompute(model.generator, criterion, model_opt), epoch, train_src)
        valid_loss = evaluate(data_gen(valid_src, valid_tgt, para['batch'], valid_tgt_ind, d_models, device), model, 
                              SimpleLossCompute(model.generator, criterion, None))
        t_loss.append(train_loss)
        v_loss.append(valid_loss)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | train loss {:5.2f} | '
              'valid loss {:5.2f} | valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                        train_loss, valid_loss, math.exp(valid_loss)))
        print('-' * 89)
        if valid_loss < best_val_loss:
            best_val_loss = valid_loss
            best_model = model
        scheduler.step()
    # save best model
    torch.save(best_model.state_dict(), para['model_path'] + '/' + para['model_name'] + '.pt')
    # save loss
    loss=tuple((t_loss, v_loss))
    with open(para['model_path'] + '/' + para['loss_name']+'.pk','wb') as file:
         pickle.dump(loss, file)
    return best_model, loss

def evaluate_model(model,TEST,tgt_vacob,d_model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_src, test_tgt, test_tgt_ind, test_total = TEST
    scores=[]
    for src, tgt, tgt_ind in zip(test_src, test_tgt, test_tgt_ind): 
        pred_tgt_ind=predict(model, src, tgt_vacob, device, d_model, 7)
        k = min(2, len(pred_tgt_ind))
        tgt = tgt.to(device)
        score = bleu(pred_tgt_ind, tgt_ind.to(device), k)
        scores.append(score)
    return np.sum(scores)/len(test_src)

def plot_loss(loss):
    t_loss, v_loss = loss
    fig = plt.figure(figsize=(8,6))
    ax1 = fig.add_subplot(111)
    plot1 = ax1.plot(t_loss, label = 'Loss_training')
    plot2 = ax1.plot(v_loss, label = 'Loss_validation')
    ax1.set_xlabel('Epoch', fontsize=20)
    ax1.set_ylabel('Loss', fontsize=20)
    lines = plot1 + plot2
    ax1.legend(lines, [l.get_label() for l in lines], bbox_to_anchor=(0.98, 0.98), fontsize=16)
    ax1.tick_params(axis='x', labelsize=16 )
    ax1.tick_params(axis='y', labelsize=16 )
    ax1.tick_params(which='major',length=4,width=1)
    ax=plt.gca()
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))
    ax.spines['bottom'].set_linewidth(1)
    ax.spines['left'].set_linewidth(1)
    ax.spines['right'].set_linewidth(1)
    ax.spines['top'].set_linewidth(1)

def greedy_decode(model, src, src_mask, tgt_vacob, device, d_model, max_len):
    memory = model.encode(src, src_mask)
    ys = torch.cat((torch.ones([1, int(d_model/2)],dtype=torch.float),
                    torch.zeros([1, int(d_model/2)],dtype=torch.float)), dim=1).type_as(src.data).unsqueeze(0)
    ys_ind = torch.ones([1],dtype=torch.int).to(device)
    for i in range(max_len-1):
        out = model.decode(memory, src_mask, 
                           Variable(ys), 
                           Variable(subsequent_mask(ys.size(1))
                                    .type_as(src.data)))
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim = 1)
       
        ys = torch.cat([ys, 
                        tgt_vacob[next_word.item()].contiguous().view(1, 1, d_model).type_as(src.data)], dim=1)
        ys_ind = torch.cat([ys_ind, next_word], dim=0)
    return ys.squeeze(), ys_ind

def predict(best_model, src, tgt_vacob, device, d_model, max_len):
    best_model.eval()
    src = src.unsqueeze(0).to(device)
    pad = torch.zeros([d_model], dtype=torch.float).to(device)
    src_mask = torch.zeros([src.shape[0], src.shape[1]],dtype=torch.bool).to(device)
    for i in range(len(src)):
        for j in range(len(src[i])):
            if src[i][j].equal(pad):
                src_mask[i][j] = False
            else:
                src_mask[i][j] = True 
    src_mask = src_mask.unsqueeze(-2)
    pred_tgt, pred_tgt_ind = greedy_decode(best_model, src, src_mask, tgt_vacob, device, d_model, max_len)
    pred_mask = torch.ones([len(pred_tgt_ind)],dtype=torch.bool).to(device)
    for i in range(len(pred_tgt_ind)):
            if pred_tgt_ind[i].item() == 0:
                pred_mask[i] = False
            if pred_tgt_ind[i].item() == 1:
                pred_mask[i] = False
            if pred_tgt_ind[i].item() == 2:
                pred_mask[i:len(pred_tgt_ind)] = False
    pred_tgt_ind = pred_tgt_ind[pred_mask]
    return pred_tgt_ind      

def bleu(pred_seq, label_seq, k):
    """BLEU"""
    len_pred, len_label = len(pred_seq), len(label_seq)
    score = math.exp(min(0, 1 - len_label / len_pred))
    for n in range(1, k + 1):
        num_matches= 0
        for i in range(len_pred - n + 1):
            for j in range(len_label - n + 1):
                if pred_seq[i: i + n].equal(label_seq[j: j + n]):
                    num_matches += 1
        score *= math.pow(num_matches / (len_pred - n + 1), math.pow(0.5, n))
    return score

def check_model(model_name, loss_name):
    model_path = f'{model_name}.pt'
    loss_path = f'{loss_name}.pk'
    return os.path.isfile(model_path) and os.path.isfile(loss_path)