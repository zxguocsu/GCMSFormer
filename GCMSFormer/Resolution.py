import numpy as np
import torch
import seaborn
seaborn.set_context(context="talk")

from tqdm import tqdm
from sklearn.metrics import explained_variance_score
from .NetCDF import netcdf_reader
from .GCMSformer import predict

import warnings
warnings.filterwarnings("ignore")


def back_remove(xx, point, range_point):

    xn = list(np.sum(xx, 1))
    n1 = xn.index(min(xn[0: range_point - point]))
    n3 = xn.index(min(xn[xx.shape[0] - range_point + point: xx.shape[0]]))
    if n1 < range_point - point / 2:
        n2 = n1 + 3
    else:
        n2 = n1 - 3
    if n3 < xx.shape[0] - range_point - point / 2:
        n4 = n3 + 3
    else:
        n4 = n3 - 3
    Ns = [[min(n1, n2), max(n1, n2)], [min(n3, n4), max(n3, n4)]]
    bak = np.zeros(xx.shape)
    for i in range(0, xx.shape[1]):
        tiab = []
        reg = []

        for j in range(0, len(Ns)):
            tt = range(Ns[j][0], Ns[j][1])
            tiab.extend(xx[tt, i])
            reg.extend(np.arange(Ns[j][0], Ns[j][1]))
        tm = tiab - np.mean(tiab)
        rm = reg - np.mean(reg)
        b = np.dot(np.dot(float(1) / np.dot(rm.T, rm), rm.T), tm)
        s = np.mean(tiab) - np.dot(np.mean(reg), b)
        b_est = s + b * np.arange(xx.shape[0])
        bak[:, i] = xx[:, i] - b_est

    bias = xx - bak
    return bak, bias


def Resolution(path, filename, model, tgt_vacob, device):
    ncr = netcdf_reader(path + '/' + filename, True)
    m = np.array(ncr.mat(0, len(ncr.tic()['rt']) - 1).T, dtype='float32')
    ms = np.array(ncr.mz_rt(10)['mz'], dtype='int')
    mz_range = (1, 1000)
    mz_min, mz_max = mz_range
    mz_dense = np.linspace(int(mz_min), int(mz_max), int(mz_max - mz_min) + 1, dtype=np.float32)
    X = np.zeros((m.shape[0], mz_max), dtype=np.float32)
    for i in range(m.shape[0]):
        itensity_dense = np.zeros_like(mz_dense)
        itensity_dense[ms - mz_min] = m[i]
        X[i] = itensity_dense
        X[i][X[i] < 0] = 0

    num = len(X) // 500
    io = np.empty(0, dtype='int32')
    for i in range(num - 1):
        X0 = X[500 * i:500 * (i + 1)]
        ind = np.arange(500 * i, 500 * (i + 1), 1, dtype='int32')
        xX, bias = back_remove(X0, 4, 10)
        xX[xX < 0] = 0
        noise = np.mean(np.sort(np.sum(xX, 1))[0:300])
        ind_0 = np.argwhere(np.sum(xX, 1) >= 3 * noise)[:, 0]
        io = np.hstack((io, ind[ind_0]))

    X0 = X[500 * (num - 1): len(X)]
    ind = np.arange(500 * (num - 1), len(X), 1, dtype='int32')
    xX, bias = back_remove(X0, 4, 10)
    xX[xX < 0] = 0
    noise = np.mean(np.sort(np.sum(xX, 1))[0:300])
    ind_1 = np.argwhere(np.sum(xX, 1) >= 3 * noise)[:, 0]
    io = np.hstack((io, ind[ind_1]))

    l = []
    ls = []
    for x in io:
        l.append(x)
        if x + 1 not in io:
            if len(l) >= 7:
                ls.append(l)
            l = []


    ls[0] = list(range(ls[0][0] - 5, ls[0][-1] + 1))
    ls[-1] = list(range(ls[-1][0], ls[-1][-1] + 6))
    for i in range(len(ls) - 1):
        if ls[i + 1][0] - ls[i][-1] >= 8:
            ls[i] = list(range(ls[i][0], ls[i][-1] + 6))
            ls[i + 1] = list(range(ls[i + 1][0] - 5, ls[i + 1][-1] + 1))

    sta_S0 = np.empty((0, 1000), dtype='float32')
    sta_C_list = []
    re_X_list = []
    sta_S_list = []
    R2s = []
    rts = []
    area = []
    RT_ind = []
    r_2_1 = []
    pred_tgt_ind_list = []
    for j in tqdm(range(len(ls)), desc=filename):
        m = np.array(ncr.mat(ls[j][0], ls[j][-1]).T, dtype='float32')
        if len(ls[j]) <= 20 and np.sum(m, 1)[0] >= 1.02 * np.sum(m, 1)[-1]:
            m0 = np.array(ncr.mat(ls[j][0] - 4, ls[j][-1] + 1).T, dtype='float32')
            t0 = np.arange(ls[j][0] - 4, ls[j][-1] + 2)
        else:
            m0 = np.array(ncr.mat(ls[j][0] + 1, ls[j][-1]).T, dtype='float32')
            t0 = np.arange(ls[j][0] + 1 , ls[j][-1] + 1)


        def converts(m):
            X = np.zeros((m.shape[0], mz_max), dtype=np.float32)
            for i in range(m.shape[0]):
                itensity_dense = np.zeros_like(mz_dense)
                itensity_dense[ms - mz_min] = m[i]
                X[i] = itensity_dense
                X[i][X[i] < 0] = 0
            xX, bias = back_remove(X, 4, 10)
            xX[xX < 0] = 0
            X = xX / np.max(xX)
            return X, xX

        X0, xX0 = converts(m0)
        new_num = []
        u, s0, v = np.linalg.svd(X0)
        for i in range(len(s0) - 1):
            if s0[i] > 0.5:
                new_num.append(s0[i])
            else:
                if s0[i] - s0[i + 1] > 0.15:
                    new_num.append(s0[i])
                else:
                    if s0[i] - s0[i + 1] > 0.08 and s0[i] > 0.15:
                        new_num.append(s0[i])

        rt = ncr.tic()['rt']
        X0 = torch.from_numpy(X0).float()
        pred_tgt_ind = predict(model, X0, tgt_vacob, device, d_model=1000, max_len=7)

        pred_tgt_ind = sorted(set(pred_tgt_ind.tolist()), key=pred_tgt_ind.tolist().index)
        if len(pred_tgt_ind) < len(new_num) and new_num[-1] > 0.2:
            m1 = m0[2:-2]
            t1 = t0[2:-2]
            X1, xX1 = converts(m1)
            new_num1 = []
            u, s0, v = np.linalg.svd(X1)

            for i in range(len(s0) - 1):
                if s0[i] > 0.5:
                    new_num1.append(s0[i])
                else:
                    if s0[i] - s0[i + 1] > 0.15:
                        new_num1.append(s0[i])
                    else:
                        if s0[i] - s0[i + 1] > 0.08 and s0[i] > 0.15:
                            new_num1.append(s0[i])

            if len(pred_tgt_ind) < len(new_num1) and new_num1[-1] > 0.2:
                pred_tgt_ind1 = pred_tgt_ind
                X_0 = X0.numpy()
                k = pred_tgt_ind1
                for t in range(len(new_num1) - len(pred_tgt_ind)):
                    X_opr = np.zeros_like(X_0)
                    P = np.zeros((1000, len(k)), dtype=np.float32)
                    for i, n in enumerate(k):
                        P[:, i] = tgt_vacob[n].numpy()
                    u, s0, v = np.linalg.svd(P.T)
                    I = np.identity(1000, dtype=np.float32)
                    Mk = I - np.dot(v[0:len(k)].T, (v[0:len(k)]))

                    for i in range(len(X_0)):
                        c = np.dot(X_0[i].reshape(1, 1000), Mk)
                        X_opr[i] = c

                    X_opr[X_opr < 0] = 0
                    X_1 = X_opr / np.max(X_opr)
                    X_opr0 = torch.from_numpy(X_1).float()
                    pred_tgt_ind0 = predict(model, X_opr0, tgt_vacob, device, d_model=1000, max_len=7)
                    pred_tgt_ind0 = sorted(set(pred_tgt_ind0.tolist()), key=pred_tgt_ind0.tolist().index)
                    if pred_tgt_ind0 == pred_tgt_ind1:
                        ind_x = list(np.where(np.sum(X_1, 1) / max(np.sum(X_1, 1)) < 0.08)[0])
                        X_opr0[ind_x] = 0
                        pred_tgt_ind0 = predict(model, X_opr0, tgt_vacob, device, d_model=1000, max_len=7)
                        pred_tgt_ind0 = sorted(set(pred_tgt_ind0.tolist()), key=pred_tgt_ind0.tolist().index)

                    pred_tgt_ind1 = pred_tgt_ind1 + pred_tgt_ind0
                    pred_tgt_ind1 = sorted(set(pred_tgt_ind1), key=pred_tgt_ind1.index)

                    if len(pred_tgt_ind1) == len(new_num1):
                        break

                    k = pred_tgt_ind1

                if len(pred_tgt_ind1) <= len(new_num1):
                    pred_tgt_ind = pred_tgt_ind1

        if len(pred_tgt_ind) == len(new_num):
            m0 = np.array(ncr.mat(ls[j][0] - 2, ls[j][-1] + 1).T, dtype='float32')
            t0 = np.arange(ls[j][0] - 2, ls[j][-1] + 2)
            X0, xX0 = converts(m0)
            X0 = torch.from_numpy(X0).float()

        S = np.zeros((len(pred_tgt_ind), 1000), dtype=np.float32)
        pred_tgt_ind_list.append(pred_tgt_ind)

        for i in range(len(pred_tgt_ind)):
            S[i] = (tgt_vacob[pred_tgt_ind[i]].cpu()).numpy()
        for i in range(0, 200):
            S[S < 0] = 0
            C = np.dot(np.dot(X0.numpy(), S.T), np.linalg.pinv(np.dot(S, S.T)))
            C[C < 0] = 0
            S = np.dot(np.dot(np.linalg.pinv(np.dot(C.T, C)), C.T), X0.numpy())
            S[S < 0] = 0
        sta_S = np.zeros_like(S)
        for i in range(len(sta_S)):
            sta_S[i] = S[i] / np.max(S[i]) * 999
        sta_C = np.dot(np.dot(xX0, sta_S.T), np.linalg.pinv(np.dot(sta_S, sta_S.T)))
        sta_C[sta_C < 0] = 0
        re_X = np.dot(sta_C, sta_S)
        R2_0 = explained_variance_score(xX0, re_X, multioutput='variance_weighted')
        R2s.append(R2_0)

        RT_ind.append(round(rt[ls[j][np.argmax(np.sum(m, 1))]], 3))


        sta_S0 = np.vstack((sta_S0, sta_S))
        sta_C_list.append(sta_C)
        re_X_list.append(re_X)
        sta_S_list.append(sta_S)


        for i in range(len(sta_S)):
            r_2_1.append(R2_0)

        for i in range(len(sta_S)):
            maxindex = np.argmax(sta_C[:, i])
            tic = ncr.tic()
            rt0 = round(tic['rt'][t0[maxindex]].astype(np.float32), 2)
            rts.append(rt0)
            compound = np.trapz(np.sum(np.dot(np.array(sta_C[:, i], ndmin=2).T, np.array(sta_S[i], ndmin=2)), 1, dtype='float32'))
            area.append(compound)

    return pred_tgt_ind_list,sta_C_list, sta_S_list, re_X_list, sta_S0, area, rts, r_2_1


def output_msp(filename_path, sta_S, RT):
    sta_S[sta_S < 3] = 0
    f = open(filename_path, "x")
    for i in range(len(sta_S)):
        f.write("Name: ")
        f.write(str(RT[i]))
        f.write('\n')
        f.write("RT: ")
        f.write(str(RT[i]))
        f.write('\n')
        f.write("Num Peaks: ")
        f.write(str(sta_S.shape[1]))
        f.write('\n')
        for n in range(sta_S.shape[1]):
            f.write(str(n + 1))
            f.write(' ')
            f.write(str(sta_S[i, n]))
            f.write('\n')
        f.write('\n')
    f.close()