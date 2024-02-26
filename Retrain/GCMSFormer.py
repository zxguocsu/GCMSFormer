import torch
import math, copy, time
import seaborn
import random
import os
seaborn.set_context(context="talk")
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick


def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

# 在data_gen里面用
class Batch:
    "Object for holding a batch of data with mask during training."
    # trg是一个batch里面的质谱（已填充对齐）
    def __init__(self, src, batch_tgt_ind, src_mask, trg=None, pad=0):
        self.src = src
        self.src_mask = src_mask
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if trg is not None:
            self.trg = trg[:, :-1, :]  # 去除每个矩阵的最后一行
            self.trg_y = trg[:, 1:, :]  # 去除每个矩阵的第一行(相当于删去bos)
            self.trg_y_ind = batch_tgt_ind[:, 1:]
            self.trg_mask = self.make_std_mask(self.trg, pad).to(device)  # 解码器输入部分掩码

            # 生成和trg_y形状相同的二维布尔矩阵  （batch_size,质谱数）
            trg_y_mask = torch.zeros([self.trg_y.shape[0], self.trg_y.shape[1]], dtype=torch.bool).to(device)

            for i in range(len(self.trg_y)):
                for j in range(len(self.trg_y[i])):
                    if self.trg_y[i][j].equal(pad):
                        trg_y_mask[i][j] = False
                    else:
                        trg_y_mask[i][j] = True
            self.ntokens = (trg_y_mask != False).data.sum()  # 对trg_y_mask中是true的元素个数求和

    # 生成decoder输入部分的掩码张量
    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        # tgt.shape[0] = batch_size大小   tgt.shape[1]] = 质谱数大小
        tgt_mask = torch.zeros([tgt.shape[0], tgt.shape[1]], dtype=torch.bool)
        for i in range(len(tgt)):  # 循环batch_size大小次
            for j in range(len(tgt[i])):  # # 循环质谱数次
                if tgt[i][j].equal(pad):
                    tgt_mask[i][j] = False
                else:
                    tgt_mask[i][j] = True
        tgt_mask = tgt_mask.unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(subsequent_mask(tgt.size(1)).type_as(tgt_mask.data))
        return tgt_mask

# 传入已经按batch_size划分好的X，S，d_model,S_index，返回补齐的数据
# 划分batch在data_gen函数的前半部分实现
# 改动了mask
def batchify(batch_source, batch_target, d_models, target_ind):
    "Convert the dataset into a small batch, filled sequence." # 将数据集转换为小批量的、填充的序列
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 将编码器输入的batch补齐并生成编码器输入的掩码
    max_source = 0

    # 遍历每个重叠峰数据，假设第一个重叠峰扫描点数为48，最大的重叠峰扫描点数为60
    for l in batch_source:
        if len(l) > max_source:
            max_source = len(l)
    # 此循环结束后，max_source的值变为60

    src_mask = torch.ones([len(batch_source), max_source], dtype=torch.bool)  # src_mask：（3,60）的全为True的矩阵

    # 对每一个扫描点个数小于60的进行扫描点填充
    # 以第一个重叠峰（扫描点48），batch_size=3 为例
    for i in range(len(batch_source)):
        pad_len = max_source - len(batch_source[i])  # pad_len = 60-48 = 12
        pad_source = torch.zeros([pad_len, d_models], dtype=torch.float)  # pad_source:(12,1000)的零矩阵
        batch_source[i] = torch.cat((batch_source[i], pad_source), dim=0)  # 将pad的扫描点直接拼接在最后一个扫描点后面
        src_mask[i][max_source - pad_len:max_source] = False  # 即将src_mask填充部分变为False

    # 相当于将输入的掩码扩充成3个 1×60 的矩阵
    src_mask = src_mask.unsqueeze(-2).to(device)  # (3,1,60) (batch_size,1,max_source)

    # 将解码器输入的batch补齐并生成解码器输入的掩码
    max_target = 0
    # 遍历每个质谱标签，假设最多重叠组分=5 第一个组分=2
    for l in batch_target:
        if len(l) > max_target:
            max_target = len(l)
    # 遍历结束后，max_target=5

    # 循环batch_size大小次 假设最多重叠组分=5 第一个组分=2
    for i in range(len(batch_target)):
        pad_traget = torch.zeros([max_target - len(batch_target[i]), d_models], dtype=torch.float)  # （5-2，1000）的零矩阵
        pad_ind = torch.zeros([max_target - len(batch_target[i])], dtype=torch.int)  # （1，5-2）的零矩阵

        # 将质谱标签补齐
        # bos = [1....,0....]
        bos = torch.cat((torch.ones([1, int(d_models / 2)], dtype=torch.float),
                         torch.zeros([1, int(d_models / 2)], dtype=torch.float)), dim=1)
        # eos = [0....,1....]
        eos = torch.cat((torch.zeros([1, int(d_models / 2)], dtype=torch.float),
                         torch.ones([1, int(d_models / 2)], dtype=torch.float)), dim=1)
        # 按顺序(bos+target+eos+pad)拼接，形成新的质谱标签
        batch_target[i] = torch.cat((bos, batch_target[i], eos, pad_traget), dim=0)

        # 将质谱索引标签补齐
        bos_ind = torch.ones([1], dtype=torch.int)  # bos的索引为1
        eos_ind = torch.LongTensor([2])  # eos的索引为2
        target_ind[i] = torch.cat((bos_ind, target_ind[i], eos_ind, pad_ind), dim=0)
        # 列表target_ind：
        # [ tensor([ 1, 31, 68, 62, 65,  2,  0]),
        #   tensor([ 1, 61, 66, 52,  2,  0,  0]),
        #   tensor([ 1,  3, 41, 32, 67, 64,  2])]



    # (batch_size,max_source+2,1000) 相当于可以从中看出batch_size的大小，以及补齐后的数据维度
    batch_source = Variable(torch.stack(batch_source, 0), requires_grad=False).to(device)
    batch_target = Variable(torch.stack(batch_target, 0), requires_grad=False).to(device)
    target_ind = Variable(torch.stack(target_ind, 0), requires_grad=False).to(device)

    # 返回结果的数据类型是一个tensor
    return batch_source, batch_target, target_ind, src_mask
    # 张量target_ind：
    # tensor([[ 1, 31, 68, 62, 65,  2,  0],
    #         [ 1, 61, 66, 52,  2,  0,  0],
    #         [ 1,  3, 41, 32, 67, 64,  2]])


# 返回迭代数据
# 在训练和验证的loss里用
def data_gen(src, tgt, batch_size, tgt_inds, d_models, device):
    "Generate batch_data for training, evaluation, and testing models."
    nbatches = len(src) // batch_size  # 数据个数//batch_size ——> 一个epoch里面的迭代次数

    # 把不能整除的部分舍掉，可以整除的部分作为模型可以使用的数据
    src = src[0:(nbatches * batch_size)]
    tgt = tgt[0:(nbatches * batch_size)]
    tgt_inds = tgt_inds[0:(nbatches * batch_size)]

    # zip() 函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表。
    # 例如：a = ['a', 'b', 'c', 'd']
    #      b = ['1', '2', '3', '4']
    #      list(zip(a, b))
    # 结果：[('a', '1'), ('b', '2'), ('c', '3'), ('d', '4')]
    data_zip = list(zip(src, tgt, tgt_inds))

    # random.shuffle()用于将一个列表中的元素打乱顺序，值得注意的是使用这个方法不会生成新的列表，只是将原列表的次序打乱。
    random.shuffle(data_zip)

    # zip（*） 解压，为zip的逆过程，相当于把原来对应打包的又拆开了
    src_rnd, tgt_rnd, tgt_inds_rnd = tuple(zip(*data_zip))
    # src_rnd, tgt_rnd, tgt_inds_rnd 的数据类型为tuple

    # 按迭代次数划分数据
    for i in range(nbatches):
        src_data = list(src_rnd[batch_size * i: batch_size * (i + 1)])
        tgt_data = list(tgt_rnd[batch_size * i: batch_size * (i + 1)])
        traget_ind = list(tgt_inds_rnd[batch_size * i: batch_size * (i + 1)])
        # src_data, tgt_data, traget_ind 的数据类型为list

        # 将数据集转换为小批量的、填充的序列
        batch_src, batch_tgt, batch_tgt_ind, src_mask = batchify(src_data, tgt_data, d_models, traget_ind)
        pad = torch.zeros([d_models], dtype=torch.float).to(device)  # 生成维度为1000的标量

        # class Batch:
        #     def __init__(self, src, batch_tgt_ind, src_mask, trg=None, pad=0):
        yield Batch(batch_src, batch_tgt_ind, src_mask, batch_tgt, pad)  # yield：返回一个可迭代的类的实例化对象


'''=============================================框架里面基本结构的构建==================================================='''
class Embeddings(nn.Module):
    def __init__(self, d_model):
        # vocab:指词表大小
        super(Embeddings, self).__init__()
        self.lut = nn.Linear(d_model, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    # 将位置编码数据直接与原数据相加
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)


def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


# 克隆模型并存放至一个列表中（列表中的每个元素都是一个模型）
def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


# 最后输出的维度（batch_size,scan_point,d_model）——> 经过线性层不改变维度？
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
        x, self.attn = attention(query, key, value, mask=mask,dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        del query
        del key
        del value
        return self.linears[-1](x)


# 考虑注意力机制可能对复杂过程的拟合程度不够，通过增加两层网络来增强模型的能力
class PositionwiseFeedForward(nn.Module):   # 前馈全连接层：具有两层线性层的全连接网络
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)  # 本质是生成一个可以将d_model个输入转化成d_ff个输出的权重矩阵
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


# 随着网络层数的增加，通过多层计算后参数可能开始出现过大或过小的情况，可能导致模型难以收敛，规范化层可以使特征数值控制在合理范围内
class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):   # features = d_model  eps:一个足够小的数，在规范化公式的分母中出现，防止分母为0
        super(LayerNorm, self).__init__()

        # nn.Parameter封装后表示他们使模型的参数（未来会随着模型一起训练）
        self.a_2 = nn.Parameter(torch.ones(features))  # 全1张量，缩放参数
        self.b_2 = nn.Parameter(torch.zeros(features))  # 全0张量，位移参数
        self.eps = eps

    def forward(self, x):
        # 首先对输入变量x求其最后一个维度的均值，并保持输出维度与输入维度一致
        mean = x.mean(-1, keepdim=True)  # keepdim=True：保持前后维度相同
        # 接着求最后一个维度的标准差
        std = x.std(-1, keepdim=True)

        # 规范化公式：（x-均值）/ 标准差
        # 为了防止标准差=0，会在标准差加一个足够小的数
        # * 表示同型点乘，即对应位置进行乘法操作
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


# 子层，包括残差连接
class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    # size：词嵌入维度的大小（ = feature = d_model）
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        # x：上一层或者子层的输入   sublayer：该子层连接中的子层函数
        # 首先将x进行规范化，然后送入子层函数中处理，处理结果进入dropout层，最后进行残差链接
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


# 编码器层
class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"

    # size = d_model  self_attn:多头自注意力子层的实例化对象  feed_forward：前馈前连接层的实例化对象
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward

        # 编码器中有两个子层结构，因此克隆两个
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        # 首先通过第一个子层，包括多头自注意力子层，然后通过第二个子层，包括前馈全连接子层
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)

# 将编码器层复制 N 层
class Encoder(nn.Module):
    "Core encoder is a stack of N layers"

    # layer:编码器层  N：编码器个数
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def get_layer_result(self,x, mask):
        memory_mid = []
        for layer in self.layers:
            x = layer(x, mask)
            memory_mid.append(self.norm(x))
        return memory_mid

    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


# 解码器层：作为解码器的组成单元，每个解码器层根据给定的输入向目标方向进行特征提取
class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    # self_attn：多头自注意力对象Q=K=V  src_attn：多头注意力对象Q!=K=V
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):  # src_mask：源数据掩码张量 tgt_mask：目标数据掩码张量
        "Follow Figure 1 (right) for connections."
        m = memory  # 来自编码器层的语义存储变量memory

        # 第一个子层的最后一个参数是目标数据掩码张量，这时要对目标数据进行遮掩，因为此时模型可能还没有生成任何目标数据
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))

        # 第二个子层中的k和v是编码层输出memory（相当于给解码器一个提示）
        # src_mask：用于遮蔽掉对结果没有意义的字符而产生的注意力值 以此提升模型模型效果和训练速度
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))

        return self.sublayer[2](x, self.feed_forward)


# 将解码器层复制 N 层
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


class Generator(nn.Module):
    "Define standard linear + softmax generation step. /nn.CrossEntropyLoss"

    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)  # 线性层的作用：转换维度

    def forward(self, x):
        return self.proj(x)
    #   return F.log_softmax(self.proj(x),dim=-1)


# 把encoder和decoder连接起来
class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """
    # src_embed:源数据的嵌入函数（输入先经过词嵌入层再经过位置编码层）
    # tgt_embed：目标数据的嵌入函数（输入先经过词嵌入层再经过位置编码层）
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def get_mid_memory(self,src, src_mask):
        return self.encoder.get_layer_result(self.src_embed(src), src_mask)

    # src：源数据
    # 使用src_embed对src做处理，然后和src_mask一起传给self.encoder
    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    # tgt：目标数据
    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

    # 将src和src_mask传入编码函数，得到结果后，与src_mask, tgt, tgt_mask一同传给解码函数

    # out = model.forward(batch.src, batch.trg, batch.src_mask, batch.trg_mask)
    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

'''-----------------------------------------训练模型----------------------------------------------------------------'''

'''===========================================用基本结构搭建模型框架=============================================='''

# 模型输出形状（batch_size,化合物数目,d_model）
def make_model(tgt_vocab, N=4, d_model=1000, d_ff=1024, h=8, dropout=0.1):
    "Helper: Construct a model from hyperparameters."

    tgt_vocab_lens = len(tgt_vocab)
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    # EncoderDecoder(encoder, decoder, src_embed, tgt_embed, generator)
    # nn.Sequential(): 模型接收的输入首先被传入nn.Sequential()包含的第一个网络模块中。
    #                  然后，第一个网络模块的输出传入第二个网络模块作为输入，按照顺序依次计算并传播，
    #                  直到nn.Sequential()里的最后一个模块输出结果。

    # 传入的都是实例化对象，此时的model是EncoderDecoder的实例化对象
    # 模型5大块：两个输入、encoder、decoder、一个输出
    model = EncoderDecoder( Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
                            Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
                            nn.Sequential(Embeddings(d_model), c(position)),
                            nn.Sequential(Embeddings(d_model), c(position)),
                            Generator(d_model, tgt_vocab_lens)
                            )

    # 模型结构完成后，接下来就是初始化模型中的参数，比如线性层中的变换矩阵
    # 这里一但判断参数维度大于1，则会将其初始化一个服从均匀分布的矩阵
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
    return model

'''=====================================定义训练和测试时候的loss,在train_model里面使用=================================='''
# 模型输出结果在计算交叉熵损失之前先经过generator转换成形状（batch_size,化合物数目,len（tgt_vocab））,再计算
class SimpleLossCompute:
    "A simple loss compute and train function. /nn.CrossEntropyLoss"

    # loss_compute = SimpleLossCompute(model.generator, criterion, None) 实例化对象
    # loss = loss_compute(out, batch.trg_y_ind)
    # out = model.forward(batch.src, batch.trg, batch.src_mask, batch.trg_mask)
    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt

    def __call__(self, x, y):
        x = self.generator(x)
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)), y.contiguous().view(-1))
        if self.opt is not None:
            loss.backward()
            self.opt.step()
            self.opt.zero_grad()
        return loss.item()

# 计算训练时候的损失
def train(data_iter, batch_size, model, loss_compute, epoch, train_src):
    # data_iter = data_gen(train_src, train_tgt, para['batch'], train_tgt_ind, d_models, device)
    # loss_compute = SimpleLossCompute(model.generator, criterion, model_opt)

    global i
    model.train()  # model.train()的作用：启用 Batch Normalization 和 Dropout
    total_loss = 0  # 记录迭代log_interval次的loss
    total_losss = 0  # 记录整个训练的loss
    start = time.time()  # 返回当前时间的时间戳

    for i, batch in enumerate(data_iter): # i是每个epoch里面的迭代次数
        # def forward(self, src, tgt, src_mask, tgt_mask):
        out = model.forward(batch.src, batch.trg, batch.src_mask, batch.trg_mask)
        loss = loss_compute(out, batch.trg_y_ind)
        total_loss += loss
        total_losss += loss
        log_interval = 1000  #20 # 50

        if i % log_interval == 0 and i > 0:  # 每次训练50组打印一次结果
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start  # 记录每迭代log_interval次需要花的时间

            print('| epoch {:3d} | {:5d}/{:5d} batches | s/batch {:5.2f} | loss {:5.2f} | ppl {:8.2f}'
            .format(epoch, i, len(train_src) // batch_size, elapsed / log_interval, cur_loss, math.exp(cur_loss)))
            # train_src 是所有用来训练的数据 ，len(train_src)就是所有用来训练的数据的个数，
            # len(train_src) // batch_size 就是需要迭代的总次数
            # elapsed / log_interval 是每迭代20次，计算每次需要花费的平均时间
            # cur_loss 是20次迭代后的平均loss

            total_loss = 0
            start = time.time()

    return total_losss / i

# 计算验证时候的损失
def evaluate(data_iter, eval_model, loss_compute):
    eval_model.eval()  # eval_model.eval()的作用：不启用 Batch Normalization 和 Dropout
    total_loss = 0
    with torch.no_grad():
        for i, batch in enumerate(data_iter):
            out = eval_model.forward(batch.src, batch.trg, batch.src_mask, batch.trg_mask)
            loss = loss_compute(out, batch.trg_y_ind)
            total_loss += loss
            del loss
    return total_loss / i

'''===============================用train data训练模型  valid data选择合适超参数  两者结合确定最终模型======================'''
# para是自己输入  TRAIN、VALID从data_split得到，是8：1：1的前两部分
# 返回best_model
def train_model(para, TRAIN, VALID, tgt_vacob):

    # load Datasets
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 训练模型在gpu
    tgt_vacob = tgt_vacob.to(device)
    d_models = int(max(para['mz_range']))

    # 将元组TRAIN和VALID里的元素分别赋值，赋值后的结果（train_src, train_tgt, train_tgt_ind, train_total）是列表
    train_src, train_tgt, train_tgt_ind, train_total = TRAIN
    valid_src, valid_tgt, valid_tgt_ind, valid_total = VALID
    best_val_loss = float("inf")  # 正无穷
    best_model = None

    # Initialization model
    model = make_model(tgt_vacob, N=para['layer_num'], h=para['head']).to(device)

    # loss function
    criterion = nn.CrossEntropyLoss()

    # optimizer and lr
    model_opt = torch.optim.AdamW(model.parameters(), lr=para['lr'])
    scheduler = torch.optim.lr_scheduler.StepLR(model_opt, 1, gamma=0.95)
    # 表示每训练1个epoch，下一次的epoch的学习率为上次的0.95倍
    # torch.optim.lr_scheduler.StepLR(optimizer, step_size, gamma=0.1, last_epoch=- 1, verbose=False)
    # step_size 表示每训练step_size个epoch，更新一次参数
    # gamma 表示更新lr的乘法因子

    # train and evaluate model 随着epoch的增加，两个loss应该都是下降
    t_loss = []  # 记录所有epoch后的train的每一个loss  比如有50个epoch，就有50个loss值
    v_loss = []  # 记录所有epoch后的valid的每一个loss

    # 每个epoch里都计算一下train和vali的loss，并进行比较
    for epoch in range(1, para['epochs'] + 1):
        epoch_start_time = time.time()  # 第n个epoch开始的时间

        # train(data_iter, batch_size, model, loss_compute, epoch, train_src)
        train_loss = train(
                            data_gen(train_src, train_tgt, para['batch_size'], train_tgt_ind, d_models, device),
                            para['batch_size'],
                            model,
                            SimpleLossCompute(model.generator, criterion, model_opt),
                            epoch,
                            train_src
                           )

        # evaluate(data_iter, eval_model, loss_compute)
        valid_loss = evaluate(
                              data_gen(valid_src, valid_tgt, para['batch_size'], valid_tgt_ind, d_models, device),
                              model,
                              SimpleLossCompute(model.generator, criterion, None)
                              )

        t_loss.append(train_loss)
        v_loss.append(valid_loss)

        print('-' * 89)
        # :8.2f 表示是输出带小数的浮点数，小数位数为两位，整个浮点数，包括小数点为八位数，不足8个字符，会在前面补充空格
        # :3d 中d表示要输出一个整数，3表示这个整数至少要占3个字符，如果这个整数只有一位数不足3个字符，会在前面补充空格
        print('| end of epoch {:3d} | time: {:5.2f}s | train loss {:5.2f} | valid loss {:5.2f} | valid ppl {:5.2f}'
              .format(epoch, (time.time() - epoch_start_time), train_loss, valid_loss, math.exp(valid_loss)))
        print('-' * 89)

        # 将每次训练好一个epoch后的模型保存下来
        if valid_loss < best_val_loss:
            best_val_loss = valid_loss
            best_model = model
        # 最后保存的是在所有epoch里，valid_loss最小的那个epoch训练的模型

        scheduler.step()  # 对优化器的学习率进行调整 ——>  每个epoch的学习率是不同的:学习率逐渐降低

    loss = tuple((t_loss, v_loss))

    return best_model, loss

# 画train_model的loss
def plot_loss(loss):
    t_loss, v_loss = loss
    fig = plt.figure(figsize=(8, 6))  # pyplot.figure() 函数的作用就是创建一个图像
    ax1 = fig.add_subplot(111)  # 111表示将画布（fig）分成1×1，然后将第一块赋值给ax1

    # 在画布绘制图画
    plot1 = ax1.plot(t_loss, label='Loss_training')  # 在ax1上绘制t_loss曲线
    plot2 = ax1.plot(v_loss, label='Loss_validation')  # 在ax1上绘制v_loss曲线

    #设置坐标轴名称
    ax1.set_xlabel('Epoch', fontsize=20)  # 设置x轴名称
    ax1.set_ylabel('Loss', fontsize=20)  # 设置y轴名称
    lines = plot1 + plot2

    # Legend图例就是为了帮助我们展示每个数据对应的图像名称
    ax1.legend(lines, [l.get_label() for l in lines], bbox_to_anchor=(0.98, 0.98), fontsize=16)
    # bbox_to_anchor：两个或四个浮点数的元组，与loc参数一起决定图例的展示位置

    # 设置刻度线
    ax1.tick_params(axis='x', labelsize=16)
    ax1.tick_params(axis='y', labelsize=16)
    ax1.tick_params(which='major', length=4, width=1)
    # 等价于ax1.tick_params(axis='both',which='major', length=4, width=1)

    # 挪动坐标轴
    ax = plt.gca()
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))
    ax.spines['bottom'].set_linewidth(1)
    ax.spines['left'].set_linewidth(1)
    ax.spines['right'].set_linewidth(1)
    ax.spines['top'].set_linewidth(1)

    plt.show()


'''==================================用valid评估模型性能 test测试泛化能力==============================================='''
# 贪婪解码是在每一步我们都选择概率最大的那个词
# 输出结果是预测出的质谱以及其对应索引（在69中的）
# 贪婪解码是在模型训练好了之后用
def greedy_decode(model, src, src_mask, tgt_vacob, device, d_model, max_len):
    memory = model.encode(src, src_mask)

    # 生成bos以及其索引
    ys = torch.cat((torch.ones([1, int(d_model / 2)], dtype=torch.float),
                    torch.zeros([1, int(d_model / 2)], dtype=torch.float)), dim=1).type_as(src.data).unsqueeze(0)
    ys_ind = torch.ones([1], dtype=torch.int).to(device)

    # 在训练时，decoder的输入时掩码后的真实标签，并且时一次全部输入decoder，而在预测时，是用for循环不断将decoder新生成的结果不断输进decoder
    # 此时的out是decoder的输出
    for i in range(max_len - 1):
        # 相当于预测时，decoder是在循环的。即第一次循环是将memory与bos输进decoder，第二次是将memory与decoder的第一次预测结果输进decoder。。。。
        out = model.decode(memory, src_mask, Variable(ys), Variable(subsequent_mask(ys.size(1)).type_as(src.data)))

        # 将最新预测出的向量（即out的最后一个向量）放进generator里面
        # out[:, -1] ：输出形状（1，预测出的质谱，d_model）里面的最后一行
        # prob是改变输出特征维度之后的结果
        prob = model.generator(out[:, -1])

        # 返回每一行的最大值, 并且会返回索引
        _, next_word = torch.max(prob, dim=1)

        # 将每次decoder的结果拼起来，重新与memory输入decoder中
        ys = torch.cat([ys, tgt_vacob[next_word.item()].contiguous().view(1, 1, d_model).type_as(src.data)], dim=1)
        ys_ind = torch.cat([ys_ind, next_word], dim=0)
        # 将句子拼接起来
    return ys.squeeze(), ys_ind

# 输出预测的质谱索引（在69中的）
# 要用到greedy_decode
def predict(best_model, src, tgt_vacob, device, d_model, max_len):  # max_len：7
    best_model.eval()
    src = src.unsqueeze(0).to(device)
    pad = torch.zeros([d_model], dtype=torch.float).to(device)
    src_mask = torch.zeros([src.shape[0], src.shape[1]], dtype=torch.bool).to(device)
    for i in range(len(src)):
        for j in range(len(src[i])):
            if src[i][j].equal(pad):
                src_mask[i][j] = False
            else:
                src_mask[i][j] = True
    src_mask = src_mask.unsqueeze(-2)
    pred_tgt, pred_tgt_ind = greedy_decode(best_model, src, src_mask, tgt_vacob, device, d_model, max_len)

    # 生成和预测出来的索引形状一样的bool矩阵
    pred_mask = torch.ones([len(pred_tgt_ind)], dtype=torch.bool).to(device)

    # 将索引为0、1、2的掩码（因为是pad、bos和eos对应的索引）
    for i in range(len(pred_tgt_ind)):
        if pred_tgt_ind[i].item() == 0:
            pred_mask[i] = False
        if pred_tgt_ind[i].item() == 1:
            pred_mask[i] = False
        if pred_tgt_ind[i].item() == 2:
            pred_mask[i:len(pred_tgt_ind)] = False
    pred_tgt_ind = pred_tgt_ind[pred_mask]
    return pred_tgt_ind

# 评估两个结果之间的相似性，用在evaluate_model里面
# 输入预测质谱索引、质谱标签索引、n—grams
def bleu(pred_seq, label_seq, k):
    """BLEU"""
    len_pred, len_label = len(pred_seq), len(label_seq)

    # score是惩罚因子
    score = math.exp(min(0, 1 - len_label / len_pred))  # 返回 e 的 x 次幂

    for n in range(1, k + 1):
        num_matches = 0
        for i in range(len_pred - n + 1):  # len_pred - n + 1 = 可以分成的组数  比如5组分按k=2可以分成4组
            for j in range(len_label - n + 1):
                if pred_seq[i: i + n].equal(label_seq[j: j + n]):
                    num_matches += 1  # num_matches:在label中匹配的次数

        # Math.pow(base，exponent) 函数返回基数（base）的指数（exponent）次幂，即 base^exponent
        # len_pred-n+1:在predict中匹配的次数
        score *= math.pow(num_matches / (len_pred - n + 1), math.pow(0.5, n))
    return score


# 加入了len_pre==0的判断条件
def bleu_new(pred_seq, label_seq, k):
    """BLEU"""
    len_pred, len_label = len(pred_seq), len(label_seq)

    # score是惩罚因子
    if len_pred == 0:
        score = 0
    else:
        score = math.exp(min(0, 1 - len_label / len_pred))  # 返回 e 的 x 次幂
        for n in range(1, k + 1):
            num_matches = 0

            # 统计pre中的每一个元素在label中出现的次数之和
            # i用来遍历pre中的元素
            for i in range(len_pred - n + 1):  # len_pred - n + 1 = 可以分成的组数  比如5组分按k=2可以分成4组
                # 看pre的第i个元素在label中出现的次数
                for j in range(len_label - n + 1):
                    if pred_seq[i: i + n].equal(label_seq[j: j + n]):
                        num_matches += 1  # num_matches:在label中匹配的次数

            # Math.pow(base，exponent) 函数返回基数（base）的指数（exponent）次幂，即 base^exponent
            # len_pred-n+1:在predict中匹配的次数(因为每个索引只出现一次，所以划分的组数和匹配数相同)
            score *= math.pow(num_matches / (len_pred - n + 1), math.pow(0.5, n))
    return score

# TEST是没有参与训练和验证的数据集，即 8：1：1的最后一个
# 要用到predict和bleu
def evaluate_model(model, TEST, tgt_vacob, d_model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_src, test_tgt, test_tgt_ind, test_total = TEST
    scores = []
    for src, tgt, tgt_ind in zip(test_src, test_tgt, test_tgt_ind):
        pred_tgt_ind = predict(model, src, tgt_vacob, device, d_model, 7)

        # 如果len(pred_tgt_ind)>2，就是2-grams  如果len(pred_tgt_ind)<2，就是1-grams
        k = min(2, len(pred_tgt_ind))
        tgt = tgt.to(device)

        # score = bleu(pred_tgt_ind, tgt_ind.to(device), k)
        score = bleu_new(pred_tgt_ind, tgt_ind.to(device), k)

        scores.append(score)
    return np.sum(scores) / len(test_src)

'''=============================================================================='''
def check_model(model_name, loss_name):
    model_path = f'{model_name}.pt'  # 格式化字符串 f'xxx{被替换字段}xxx'
    loss_path = f'{loss_name}.pk'
    return os.path.isfile(model_path) and os.path.isfile(loss_path) # os.path.isfile(path)用于检查指定的路径是否是现有的常规文件

