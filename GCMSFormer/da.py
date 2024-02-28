import numpy as np
import random
from tqdm import tqdm
from scipy import stats
import torch
import pickle


def data_pre(mss, max_mz, maxn, noise):
    "A sample data: R"
    global x0, means, sigmas, nums, ratios, a, b
    if maxn == 5:
        L0 = [random.uniform(0.115, 0.125),
              random.uniform(0.125, 0.15),
              random.uniform(0.15, 0.175),
              random.uniform(0.175, 0.2)]
        L = [random.uniform(0.115, 0.125)]
        for i in range(3):
            L.append(random.choice(L0))
        random.shuffle(L)
        m1 = random.randint(100, 300) * 0.001
        m2 = m1 + L[0]
        m3 = m2 + L[1]
        m4 = m3 + L[2]
        m5 = m4 + L[3]
        means1 = [m1, m2, m3, m4, m5]
        sigmas = random.randint(8, 14) * 0.01
        ratios = [random.uniform(0.3, 2), random.uniform(0.5, 2), random.uniform(0.5, 2), random.uniform(0.5, 2),
                  random.uniform(0.3, 2)]
        nums = random.randint(40, 100)
        if sigmas <= 0.1:
            means = means1
            a, b = -0.4, 1.5
            x0 = np.linspace(a, b, nums)
        else:
            means = means1
            a, b = -0.4, 1.5
            x0 = np.linspace(a, b, nums)
    if maxn == 4:
        L0 = [random.uniform(0.115, 0.145),
              random.uniform(0.145, 0.175),
              random.uniform(0.175, 0.2)]
        L = [random.uniform(0.115, 0.125)]
        for i in range(2):
            L.append(random.choice(L0))
        random.shuffle(L)
        m1 = random.randint(200, 280) * 0.001
        m2 = m1 + L[0]
        m3 = m2 + L[1]
        m4 = m3 + L[2]
        means1 = [m1, m2, m3, m4]
        sigmas = random.randint(8, 14) * 0.01
        ratios = [random.uniform(0.3, 2), random.uniform(0.5, 2), random.uniform(0.5, 2), random.uniform(0.3, 2)]
        nums = random.randint(30, 90)
        if sigmas <= 0.1:
            means = means1
            a, b = -0.2, 1.5
            x0 = np.linspace(a, b, nums)
        else:
            means = means1
            a, b = -0.3, 1.5
            x0 = np.linspace(a, b, nums)
    if maxn == 3:
        L0 = [random.uniform(0.115, 0.125),
              random.uniform(0.125, 0.15)]
        L = [random.uniform(0.115, 0.125)]
        for i in range(1):
            L.append(random.choice(L0))
        random.shuffle(L)
        m1 = random.randint(200, 280) * 0.001
        m2 = m1 + L[0]
        m3 = m2 + L[1]
        means1 = [m1, m2, m3]
        sigmas = random.randint(8, 14) * 0.01
        ratios = [random.uniform(0.5, 2), random.uniform(0.5, 2), random.uniform(0.5, 2)]
        nums = random.randint(20, 80)
        if sigmas <= 0.1:
            means = means1
            a, b = -0.2, 1.4
            x0 = np.linspace(a, b, nums)
        else:
            means = means1
            a, b = -0.5, 1.6
            x0 = np.linspace(a, b, nums)
    if maxn == 2:
        L = [random.uniform(0.115, 0.125)]
        m1 = random.randint(200, 280) * 0.001
        m2 = m1 + L[0]
        means1 = [m1, m2]
        sigmas = random.randint(8, 14) * 0.01
        maxn = 2
        ratios = [random.uniform(0.3, 2), random.uniform(0.3, 2)]
        nums = random.randint(20, 70)
        if sigmas <= 0.1:
            means = means1
            a, b = -0.2, 1.2
            x0 = np.linspace(a, b, nums)
        else:
            means = means1
            a, b = -0.3, 1.55
            x0 = np.linspace(a, b, nums)
    if maxn == 1:
        m1 = random.randint(200, 280) * 0.001
        means1 = [m1]
        sigmas = random.randint(8, 14) * 0.01
        maxn = 1
        ratios = [random.uniform(0.3, 2)]
        nums = random.randint(20, 70)
        if sigmas <= 0.1:
            means = means1
            a, b = -0.2, 1.2
            x0 = np.linspace(a, b, nums)
        else:
            means = means1
            a, b = -0.3, 1.55
            x0 = np.linspace(a, b, nums)

    ids4p = random.sample(range(0, len(mss)), maxn)

    S = np.zeros((max_mz, maxn), dtype=np.float32)
    C = np.zeros((nums, maxn), dtype=np.float32)
    for i in range(maxn):
        C[:, i] = stats.norm.pdf(x0, means[i], sigmas) * ratios[i]
        S[:, i] = mss[ids4p[i]] / np.sum(mss[ids4p[i]])
    X_0 = np.dot(C, S.T)

    para = np.random.randn(nums, max_mz)
    E0 = para - np.min(para)
    E = E0 * (np.max(np.sum(X_0, 1)) * noise / (E0.sum() / nums))
    X = np.float32((np.dot(C, S.T) + E) / np.max(np.dot(C, S.T) + E))
    total = ratios + means + [sigmas, a, b, nums, maxn]
    return X, X_0, S, ids4p, total


def update_lib(lib_path_pk, added_path_msp, new_lib_path_pk=None):

    with open(lib_path_pk, 'rb') as file:
        lib_dic = pickle.load(file)

    with open(added_path_msp, 'r') as file1:
        f1 = file1.read()

    file.close()
    file1.close()

    a = f1.split('\n')
    m = []
    n = []
    for i, l in enumerate(a):
        if l.startswith('Name'):
            m.append(i)
        if l.startswith('Num Peaks'):
            n.append(i)
    for t in range(len(m) - 1):
        mzs = []
        ins = []
        for j in range(n[t] + 1, m[t + 1] - 1):
            ps = a[j].split('\n ')
            ps = [p for p in ps if len(p) > 0]  #
            for p in ps:
                mz_in = p.split(' ')
                mzs.append(int(float(mz_in[0])))
                ins.append(np.float32((mz_in[1])))
        ms_added_dic = {'m/z': mzs, 'intensity': ins}
        ms_dic = {'ms': ms_added_dic}
        name = a[m[t]].split(':')[1].strip()
        lib_dic.update({f'{name}': ms_dic})

    mzs = []
    ins = []
    for j in range(n[-1] + 1, len(a)):
        ps = a[j].split('\n ')
        ps = [p for p in ps if len(p) > 0]  #
        for p in ps:
            mz_in = p.split(' ')
            mzs.append(int(float(mz_in[0])))
            ins.append(np.float32((mz_in[1])))
    ms_added_dic = {'m/z': mzs, 'intensity': ins}
    ms_dic = {'ms': ms_added_dic}
    name = a[m[-1]].split(':')[1].strip()
    lib_dic.update({f'{name}': ms_dic})
    if new_lib_path_pk:
        with open(new_lib_path_pk, "wb") as file2:
            pickle.dump(lib_dic, file2, protocol=pickle.HIGHEST_PROTOCOL)
        print('The new library has been successfully saved as a Pickle file')
    file2.close()


def add_msp_MZmine(lib_path_msp, added_path_msp):
    with open(added_path_msp, 'r') as file1:
        content = file1.read()

    with open(lib_path_msp, 'a') as file2:
        file2.write(content)

    file1.close()
    file2.close()


def read_msp_MZmine(msp_file_path, d_models):
    f = open(msp_file_path).read()
    a = f.split('\n')
    m = []
    n = []
    for i, l in enumerate(a):
        if l.startswith('Name'):
            m.append(i)
        if l.startswith('Num Peaks'):
            n.append(i)
    mss = np.zeros((0, 1000), dtype=np.float32)
    for t in range(len(m) - 1):
        mzs = []
        ins = []
        for j in range(n[t] + 1, m[t + 1] - 1):
            ps = a[j].split('\n ')
            ps = [p for p in ps if len(p) > 0]
            for p in ps:
                mz_in = p.split(' ')
                mzs.append(int(float(mz_in[0])))
                ins.append(np.float32((mz_in[1])))

        ms = np.zeros((1, 1000), dtype=np.float32)
        for i, mz in enumerate(mzs):
            ms[0, mz - 1] = ins[i]
        mss = np.vstack((mss, ms / np.max(ms)))

    mzs = []
    ins = []
    for j in range(n[-1] + 1, len(a)):
        ps = a[j].split('\n ')
        ps = [p for p in ps if len(p) > 0]
        for p in ps:
            mz_in = p.split(' ')
            mzs.append(int(float(mz_in[0])))
            ins.append(float(mz_in[1]))
    ms = np.zeros((1, 1000), dtype=np.float32)
    for i, mz in enumerate(mzs):
        ms[0, mz - 1] = ins[i]
    mss = np.vstack((mss, ms / np.max(ms)))

    RT = []
    for t in range(len(m)):
        ps = a[m[t] + 2].split('\n ')
        ps = [p for p in ps if len(p) > 0]
        for p in ps:
            mz_rt = p.split(' ')
            RT.append(np.float32((mz_rt[1])))
    return RT, mss

def data_augmentation(msp_file_path, d_models, n, noise_level=0.001):

    "Obtain datasets by data augmentation."
    RT, mss = read_msp_MZmine(msp_file_path, d_models)
    DATA = []
    TARGET = []
    TOTAL = []
    for i in tqdm(range(n), desc='Generating Dataset'):
        noise = random.randint(1, 50) * noise_level
        maxn = random.randint(1, 5)
        data, data_0, target, ids4p, totals = data_pre(mss, d_models, maxn, noise)
        DATA.append(torch.tensor(data))
        TARGET.append(torch.tensor(target.T))
        TOTAL.append(torch.tensor(totals))
    ms = np.zeros_like(mss)
    for i in range(len(mss)):
        ms[i] = mss[i] / np.sum(mss[i])
    tgt_vocab = torch.tensor(ms, dtype=torch.float)
    bos = torch.cat((torch.ones([1, int(d_models / 2)], dtype=torch.float),
                     torch.zeros([1, int(d_models / 2)], dtype=torch.float)), dim=1)
    eos = torch.cat((torch.zeros([1, int(d_models / 2)], dtype=torch.float),
                     torch.ones([1, int(d_models / 2)], dtype=torch.float)), dim=1)
    pad = torch.zeros([1, d_models], dtype=torch.float)
    tgt_vocab = torch.cat((pad, bos, eos, tgt_vocab), dim=0)
    TARGET_ind = []
    for t in TARGET:
        ind = []
        for i in range(len(t)):
            for j in range(len(tgt_vocab)):
                if t[i].equal(tgt_vocab[j]):
                    ind.append(j)
        TARGET_ind.append(torch.tensor(ind))

    return DATA, TARGET, tgt_vocab, TARGET_ind, TOTAL

def data_split(aug_num, d_models, msp_file_path, validation_split):

    DATA, TARGET, tgt_vocab, TARGET_ind, TOTAL = data_augmentation(msp_file_path, d_models, aug_num, noise_level=0.001)
    train_src = DATA[0:round((1 - validation_split) * aug_num)]
    train_tgt = TARGET[0:round((1 - validation_split) * aug_num)]
    train_tgt_ind = TARGET_ind[0:round((1 - validation_split) * aug_num)]
    train_total = TOTAL[0:round((1 - validation_split) * aug_num)]
    TRAIN = tuple((train_src, train_tgt, train_tgt_ind, train_total))

    valid_src = DATA[round((1 - validation_split) * aug_num):round(0.9 * aug_num)]
    valid_tgt = TARGET[round((1 - validation_split) * aug_num):round(0.9 * aug_num)]
    valid_tgt_ind = TARGET_ind[round((1 - validation_split) * aug_num):round(0.9 * aug_num)]
    valid_total = TOTAL[round((1 - validation_split) * aug_num):round(0.9 * aug_num)]
    VALID = tuple((valid_src, valid_tgt, valid_tgt_ind, valid_total))

    test_src = DATA[round(0.9 * aug_num):aug_num]
    test_tgt = TARGET[round(0.9 * aug_num):aug_num]
    test_tgt_ind = TARGET_ind[round(0.9 * aug_num):aug_num]
    test_total = TOTAL[round(0.9 * aug_num):aug_num]
    TEST = tuple((test_src, test_tgt, test_tgt_ind, test_total))

    return TRAIN, VALID, TEST, tgt_vocab

def gen_datasets(para):
    aug_nums = para['aug_num']
    validation_split = 0.2
    msp_file_path = para['name']
    d_models = int(max(para['mz_range']))
    TRAIN, VALID, TEST, tgt_vocab = data_split(aug_nums, d_models, msp_file_path, validation_split)
    return TRAIN, VALID, TEST, tgt_vocab
