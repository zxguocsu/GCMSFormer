# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 22:25:53 2022

@author: fanyingjie
"""
import numpy as np
import random
from tqdm import tqdm
from scipy import stats
import torch
from torch.autograd import Variable

def data_pre1(mss, p, maxn, noise):
    "A sample data: X = np.dot(C,S.T) (maxn, noise, scans, ratio, intensity, W, R)"
    if maxn ==5:
        m1=random.randint(-20,300)*0.001
        m2=random.randint(340,500)*0.001
        m3=random.randint(690,710)*0.001
        m4=random.randint(900,1060)*0.001
        m5=random.randint(1100,1420)*0.001
        means1 = [m1,m2,m3,m4,m5]
        sigmas = random.randint(8,14)*0.01
        ratios = [random.uniform(0.2,2), random.uniform(0.2,2), random.uniform(0.2,2), random.uniform(0.2,2), random.uniform(0.2,2)]
        nums = random.randint(40, 100)
        if sigmas <= 0.1:
            means = means1
            a,b=-0.5, 2
            x0 = np.linspace(a, b, nums)
        else:
            means = means1
            a,b=-0.7, 2
            x0 = np.linspace(a, b, nums)
    if maxn ==4:
        m1=random.randint(200,280)*0.001
        m2=random.randint(520,600)*0.001
        m3=random.randint(800,880)*0.001
        m4=random.randint(1120,1240)*0.001
        means1 = [m1,m2,m3,m4]
        sigmas = random.randint(8,14)*0.01
        ratios = [random.uniform(0.2,2), random.uniform(0.2,2), random.uniform(0.2,2), random.uniform(0.2,2)]
        nums = random.randint(30, 90)
        if sigmas <= 0.1:
            means = means1
            a,b=-0.4, 1.7
            x0 = np.linspace(a, b, nums)
        else:
            means = means1
            a,b=-0.5, 2
            x0 = np.linspace(a, b, nums)
    if maxn ==3:
        m1=random.randint(240,400)*0.001
        m2=random.randint(590,610)*0.001
        m3=random.randint(800,960)*0.001
        means1 = [m1,m2,m3]
        m4=random.randint(60,300)*0.001
        m5=random.randint(590,610)*0.001
        m6=random.randint(900,1000)*0.001
        means2 = [m4,m5,m6]
        sigmas = random.randint(8,14)*0.01
        ratios = [random.uniform(0.2,2), random.uniform(0.2,2), random.uniform(0.2,2)]
        nums = random.randint(20, 80)
        if sigmas <= 0.1:
            means = means1
            a,b=-0.2,1.4
            x0 = np.linspace(a, b, nums)
        else:
            means = means2
            a,b=-0.5,1.6
            x0 = np.linspace(a, b, nums)
    if maxn == 2:
        m1=random.randint(420,500)*0.001
        m2=random.randint(700,780)*0.001
        means1 = [m1,m2]
        m3=random.randint(330,450)*0.001
        m4=random.randint(750,870)*0.001
        means2 = [m3,m4]
        sigmas = random.randint(8,14)*0.01
        maxn = 2
        ratios = [random.uniform(0.2,2), random.uniform(0.2,2)]
        nums = random.randint(20, 70)
        if sigmas <= 0.1:
            means = means1
            a,b=-0,1.2
            x0 = np.linspace(a, b, nums)
        else:
            means = means2
            a,b=-0.3,1.55
            x0 = np.linspace(a, b, nums)
    if maxn == 1:
        m1=random.randint(420,800)*0.001
        means1 = [m1]
        sigmas = random.randint(8,14)*0.01
        maxn = 1
        ratios = [random.uniform(0.2,2)]
        nums = random.randint(20, 50)
        if sigmas <= 0.1:
            means = means1
            a,b=-0,1.3
            x0 = np.linspace(a, b, nums)
        else:
            means = means1
            a,b=-0.3,1.55
            x0 = np.linspace(a, b, nums)
    ids4p = random.sample(range(0, len(mss)), maxn)
    S = np.zeros((p, maxn), dtype = np.float32)
    C = np.zeros((nums, maxn), dtype = np.float32)
    for i in range(maxn):
        C[:,i] = stats.norm.pdf(x0, means[i], sigmas)*ratios[i]
        S[:,i] = mss[ids4p[i]]/np.sum(mss[ids4p[i]])
    X_0 = np.dot(C,S.T)
    para  = np.random.randn(nums, p)
    E0 = para-np.min(para)
    E = E0*(np.max(np.sum(X_0, 1))*noise/(E0.sum()/nums))
    X = np.float32((np.dot(C,S.T) + E)/np.max(np.dot(C,S.T) + E))
    total=ratios+means+[sigmas,a,b,nums,maxn]
    return X, X_0, S, ids4p, total

def data_pre2(mss, p, maxn, noise):
    "A sample data: R"
    if maxn ==5:
        L0=[random.uniform(0.115,0.125),
           random.uniform(0.125,0.15),
           random.uniform(0.15,0.175),
           random.uniform(0.175,0.2)]
        L=[random.uniform(0.115,0.125)]
        for i in range(3):
            L.append(random.choice(L0))
        random.shuffle(L)
        m1=random.randint(100,300)*0.001
        m2=m1+L[0]
        m3=m2+L[1]
        m4=m3+L[2]
        m5=m4+L[3]
        means1 = [m1,m2,m3,m4,m5]
        sigmas = random.randint(8,14)*0.01
        ratios = [random.uniform(0.3,2), random.uniform(0.5,2), random.uniform(0.5,2), random.uniform(0.5,2), random.uniform(0.3,2)]      
        nums = random.randint(40, 100)
        if sigmas <= 0.1:
            means = means1
            a,b=-0.4, 1.5
            x0 = np.linspace(a, b, nums)
        else:
            means = means1
            a,b=-0.4, 1.5
            x0 = np.linspace(a, b, nums)
    if maxn ==4:
        L0=[random.uniform(0.115,0.145),
           random.uniform(0.145,0.175),
           random.uniform(0.175,0.2)]
        L=[random.uniform(0.115,0.125)]
        for i in range(2):
            L.append(random.choice(L0))
        random.shuffle(L)
        m1=random.randint(200,280)*0.001
        m2=m1+L[0]
        m3=m2+L[1]
        m4=m3+L[2]
        means1 = [m1,m2,m3,m4]
        sigmas = random.randint(8,14)*0.01        
        ratios = [random.uniform(0.3,2), random.uniform(0.5,2), random.uniform(0.5,2), random.uniform(0.3,2)]
        nums = random.randint(30, 90)
        if sigmas <= 0.1:
            means = means1
            a,b=-0.2, 1.5
            x0 = np.linspace(a, b, nums)
        else:
            means = means1
            a,b=-0.3, 1.5
            x0 = np.linspace(a, b, nums)
    if maxn ==3:
        L0=[random.uniform(0.115,0.125),
           random.uniform(0.125,0.15)]
        L=[random.uniform(0.115,0.125)]
        for i in range(1):
            L.append(random.choice(L0))
        random.shuffle(L)
        m1=random.randint(200,280)*0.001
        m2=m1+L[0]
        m3=m2+L[1]
        means1 = [m1,m2,m3]
        sigmas = random.randint(8,14)*0.01
        ratios = [random.uniform(0.5,2), random.uniform(0.5,2), random.uniform(0.5,2)]
        nums = random.randint(20, 80)
        if sigmas <= 0.1:
            means = means1
            a,b=-0.2,1.4
            x0 = np.linspace(a, b, nums)
        else:
            means = means1
            a,b=-0.5,1.6
            x0 = np.linspace(a, b, nums)
    if maxn == 2:
        L=[random.uniform(0.115,0.125)]
        m1=random.randint(200,280)*0.001
        m2=m1+L[0]
        means1 = [m1,m2]
        sigmas = random.randint(8,14)*0.01
        maxn = 2
        ratios = [random.uniform(0.3,2), random.uniform(0.3,2)]
        nums = random.randint(20, 70)
        if sigmas <= 0.1:
            means = means1
            a,b=-0.2,1.2
            x0 = np.linspace(a, b, nums)
        else:
            means = means1
            a,b=-0.3,1.55
            x0 = np.linspace(a, b, nums)
    ids4p = random.sample(range(0, len(mss)), maxn)    
    S = np.zeros((p, maxn), dtype = np.float32)
    C = np.zeros((nums, maxn), dtype = np.float32)
    for i in range(maxn):
        C[:,i] = stats.norm.pdf(x0, means[i], sigmas)*ratios[i]
        S[:,i] = mss[ids4p[i]]/np.sum(mss[ids4p[i]])
    X_0 = np.dot(C,S.T)
    para  = np.random.randn(nums, p)
    E0 = para-np.min(para)
    E = E0*(np.max(np.sum(X_0, 1))*noise/(E0.sum()/nums))
    X = np.float32((np.dot(C,S.T) + E)/np.max(np.dot(C,S.T) + E))
    total=ratios+means+[sigmas,a,b,nums,maxn]
    return X, X_0, S, ids4p, total

def read_msp_MZmine(msp_file,max_mz):
    f = open(msp_file).read()
    a=f.split('\n')
    m=[]
    n=[]
    for i,l in enumerate(a):
        if l.startswith('Name'):
            m.append(i)
        if l.startswith('Num Peaks'):
            n.append(i)
    mss=np.zeros((0,1000),dtype= np.float32)
    for t in range(len(m)-1):
        mzs=[]
        ins=[]
        for j in range(n[t]+1,m[t+1]-1):
            ps=a[j].split('\n ')
            ps=[p for p in ps if len(p)>0]
            for p in ps:
                mz_in = p.split(' ')
                mzs.append(int(float(mz_in[0])))
                ins.append(np.float32((mz_in[1])))
        ms=np.zeros((1,1000),dtype= np.float32)
        for i,mz in enumerate(mzs):
            ms[0,mz-1]=ins[i]
        mss=np.vstack((mss,ms/np.max(ms)))
    mzs=[]
    ins=[]
    for j in range(n[-1]+1,len(a)):
        ps=a[j].split('\n ')
        ps=[p for p in ps if len(p)>0]
        for p in ps:
            mz_in = p.split(' ')
            mzs.append(int(float(mz_in[0])))
            ins.append(float(mz_in[1]))
    ms=np.zeros((1,1000),dtype= np.float32)
    for i,mz in enumerate(mzs):
        ms[0,mz-1]=ins[i]
    mss=np.vstack((mss,ms/np.max(ms))) 
    
    RT=[]
    for t in range(len(m)):
        ps=a[m[t]+2].split('\n ')
        ps=[p for p in ps if len(p)>0]
        for p in ps:
            mz_rt = p.split(' ')
            RT.append(np.float32((mz_rt[1])))
    return RT, mss

def data_augmentation(dbname, d_model, n, noise_level):
    "Obtain datasets by data augmentation."
    RT,mss = read_msp_MZmine(dbname, d_model)
    DATA = []
    TARGET = []
    TOTAL = []
    for i in tqdm(range(n), desc='Generating Dataset'):
        noise = random.randint(1,50) * noise_level
        if i<=int((2/3)*n):
            maxn = random.randint(1,5)
            data, data_0, target, ids4p, totals = data_pre1(mss, d_model, maxn, noise)
        else:
            maxn = random.randint(2,5)
            data, data_0, target, ids4p, totals = data_pre2(mss, d_model, maxn, noise)
        DATA.append(torch.tensor(data))
        TARGET.append(torch.tensor(target.T))
        TOTAL.append(torch.tensor(totals))
    ms = np.zeros_like(mss) 
    for i in range(len(mss)):
        ms[i]=mss[i]/np.sum(mss[i])
    tgt_vacob = torch.tensor(ms, dtype=torch.float)
    bos = torch.cat((torch.ones([1, int(d_model/2)],dtype=torch.float),
                    torch.zeros([1, int(d_model/2)],dtype=torch.float)), dim=1)
    eos = torch.cat((torch.zeros([1, int(d_model/2)],dtype=torch.float),
                    torch.ones([1, int(d_model/2)],dtype=torch.float)), dim=1)
    pad = torch.zeros([1, d_model],dtype=torch.float)
    tgt_vacob = torch.cat((pad, bos, eos, tgt_vacob), dim=0)
    TARGET_ind = []
    for t in TARGET:
        ind=[]
        for i in range(len(t)):
            for j in range(len(tgt_vacob)):
               if t[i].equal(tgt_vacob[j]):
                   ind.append(j)
        TARGET_ind.append(torch.tensor(ind))
    return DATA, TARGET, tgt_vacob, TARGET_ind, TOTAL  

def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

class Batch:
    "Object for holding a batch of data with mask during training."
    def __init__(self, src, batch_tgt_ind, src_mask, trg=None, pad=0):
        self.src = src
        self.src_mask = src_mask
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if trg is not None:
            self.trg = trg[:, :-1, :]
            self.trg_y = trg[:, 1:, :]
            self.trg_y_ind = batch_tgt_ind[:, 1:]
            self.trg_mask = \
                self.make_std_mask(self.trg, pad).to(device)
            trg_y_mask = torch.zeros([self.trg_y.shape[0], self.trg_y.shape[1]],dtype=torch.bool).to(device)
            for i in range(len(self.trg_y)):
                for j in range(len(self.trg_y[i])):
                    if self.trg_y[i][j].equal(pad):
                        trg_y_mask[i][j] = False
                    else:
                        trg_y_mask[i][j] = True
            self.ntokens = (trg_y_mask != False).data.sum()
    
    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = torch.zeros([tgt.shape[0], tgt.shape[1]],dtype=torch.bool)
        for i in range(len(tgt)):
            for j in range(len(tgt[i])):
                if tgt[i][j].equal(pad):
                    tgt_mask[i][j] = False
                else:
                    tgt_mask[i][j] = True 
        tgt_mask = tgt_mask.unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(
            subsequent_mask(tgt.size(1)).type_as(tgt_mask.data))
        return tgt_mask

def batchify(batch_source, batch_traget, d_model, traget_ind):
    "Convert the dataset into a small batch, filled sequence."
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    max_source = 0
    for l in batch_source:
        if len(l) > max_source:
            max_source = len(l)
    for i in range(len(batch_source)):
        pad_len = max_source-len(batch_source[i])
        pad_source = torch.zeros([pad_len, d_model],dtype=torch.float)
        batch_source[i] = torch.cat((batch_source[i], pad_source), dim=0)
        src_mask = torch.ones([len(batch_source), max_source],dtype=torch.bool)
        src_mask[i][max_source-pad_len:max_source] = False
    src_mask = src_mask.unsqueeze(-2).to(device)   
    max_traget = 0
    for l in batch_traget:
        if len(l) > max_traget:
            max_traget = len(l)
    for i in range(len(batch_traget)):
        pad_traget = torch.zeros([max_traget-len(batch_traget[i]), d_model],dtype=torch.float)
        pad_ind = torch.zeros([max_traget-len(batch_traget[i])],dtype=torch.int)
        bos = torch.cat((torch.ones([1, int(d_model/2)],dtype=torch.float),
                        torch.zeros([1, int(d_model/2)],dtype=torch.float)), dim=1)
        eos = torch.cat((torch.zeros([1, int(d_model/2)],dtype=torch.float),
                        torch.ones([1, int(d_model/2)],dtype=torch.float)), dim=1)
        batch_traget[i] = torch.cat((bos, batch_traget[i], eos, pad_traget), dim=0)
        bos_ind = torch.ones([1],dtype=torch.int)
        eos_ind = torch.LongTensor([2])
        traget_ind[i] = torch.cat((bos_ind, traget_ind[i], eos_ind, pad_ind), dim=0)
    batch_source =  Variable(torch.stack(batch_source, 0), 
                         requires_grad=False).to(device)
    batch_traget =  Variable(torch.stack(batch_traget, 0), 
                         requires_grad=False).to(device)
    traget_ind = Variable(torch.stack(traget_ind, 0), 
                         requires_grad=False).to(device)
    return batch_source, batch_traget, traget_ind, src_mask

def data_gen(src, tgt, batch_size, tgt_inds, d_model, device):
    "Generate batch_data for training, evaluation, and testing models."
    nbatches = len(src) // batch_size
    src = src[0:(nbatches * batch_size)]
    tgt = tgt[0:(nbatches * batch_size)]
    tgt_inds = tgt_inds[0:(nbatches * batch_size)]
    data_zip = list(zip(src, tgt, tgt_inds))
    random.shuffle(data_zip)
    src_rnd, tgt_rnd, tgt_inds_rnd = tuple(zip(*data_zip))
    for i in range(nbatches):
        src_data = list(src_rnd[batch_size*i: batch_size*(i+1)])
        tgt_data = list(tgt_rnd[batch_size*i: batch_size*(i+1)])
        traget_ind= list(tgt_inds_rnd[batch_size*i: batch_size*(i+1)])
        batch_src, batch_tgt, batch_tgt_ind, src_mask = batchify(src_data, tgt_data, d_model, traget_ind)
        pad = torch.zeros([d_model], dtype=torch.float).to(device)
        yield Batch(batch_src, batch_tgt_ind, src_mask, batch_tgt, pad)#yield 

def data_split(aug_num, d_model, dbname, validation_split):
    "Datasets split=8:1:1."
    DATA, TARGET, tgt_vacob, TARGET_ind, TOTAL = data_augmentation(dbname, d_model, aug_num, 
                                                          noise_level=0.001)
    train_src = DATA[0:round((1-validation_split)*aug_num)]
    train_tgt = TARGET[0:round((1-validation_split)*aug_num)]
    train_tgt_ind = TARGET_ind[0:round((1-validation_split)*aug_num)]
    train_total = TOTAL[0:round((1-validation_split)*aug_num)]
    TRAIN = tuple((train_src, train_tgt, train_tgt_ind, train_total))
    valid_src = DATA[round((1-validation_split)*aug_num):round(0.9*aug_num)]
    valid_tgt = TARGET[round((1-validation_split)*aug_num):round(0.9*aug_num)]
    valid_tgt_ind = TARGET_ind[round((1-validation_split)*aug_num):round(0.9*aug_num)]
    valid_total = TOTAL[round((1-validation_split)*aug_num):round(0.9*aug_num)]
    VALID = tuple((valid_src, valid_tgt, valid_tgt_ind, valid_total))
    test_src = DATA[round(0.9*aug_num):aug_num]
    test_tgt = TARGET[round(0.9*aug_num):aug_num]
    test_tgt_ind = TARGET_ind[round(0.9*aug_num):aug_num]
    test_total = TOTAL[round(0.9*aug_num):aug_num]
    TEST = tuple((test_src, test_tgt, test_tgt_ind, test_total))
    return TRAIN, VALID, TEST, tgt_vacob

def gen_datasets(para):
    aug_nums = para['aug_num']
    validation_split = 0.2
    name = para['name']
    d_models = int(max(para['mz_range']))
    TRAIN, VALID, TEST, tgt_vacob = data_split(aug_nums, d_models, name, validation_split)
    return TRAIN, VALID, TEST, tgt_vacob