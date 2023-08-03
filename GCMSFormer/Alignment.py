import numpy as np
import seaborn
seaborn.set_context(context="talk")
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from .Resolution import Resolution, output_msp
import warnings
warnings.filterwarnings("ignore")


def Alignment(path,files, model,tgt_vacob,device):
    T = []
    S = []
    A = []
    R = []
    N = []
    for filename in files:
        pred_tgt_ind_list,sta_C_list, sta_S_list, re_X_list, sta_S, area, rt, R2 =  Resolution(path,filename,model,tgt_vacob,device)
        msp = filename.split('.CDF')[0] + '.MSP'
        output_msp(path + '/'+ msp, sta_S, rt)
        T.append(rt)
        S.append(sta_S)
        A.append(area)
        R.append(R2)
        N.append((np.vstack((np.array(rt),np.array(area),np.array(R2)))).T)  # np.vstack（）返回竖直堆叠后的数组
    T_list = np.array(T[0])
    S_list = np.array(S[0])
    A_list = np.array(A[0])
    R_list = np.array(R[0])
    for i in range(1, len(S)):
        S_list = np.vstack((S_list, np.array(S[i])))
        T_list = np.hstack((T_list, np.array(T[i])))
        A_list = np.hstack((A_list, np.array(A[i])))
        R_list = np.hstack((R_list, np.array(R[i])))
    index_list = np.argsort(T_list)
    index_rt = np.sort(T_list)
    l  = [index_list[0]]
    ls_ind = []
    for i in range(1, len(index_rt)):
        if index_rt[i]-index_rt[i-1] <= 0.1:
            l.append(index_list[i])
        else:
            ls_ind.append(l)
            l = [index_list[i]]
            if i == len(index_rt)-1:
                ls_ind.append(l)
    l = [index_rt[0]]
    ls_rt = []
    for i in range(1, len(index_rt)):
        if index_rt[i]-index_rt[i-1] <= 0.1:
            l.append(index_rt[i])
        else:
            ls_rt.append(l)
            l = [index_rt[i]]
            if i == len(index_rt)-1:
                ls_rt.append(l)
    lst_ind = []
    lst_rt = []
    tr0 = []
    tr1 = []
    tr2 = []
    tr3 = []
    for i in range(len(ls_ind)):
        while len(ls_ind[i]) > 0:
            if len(ls_ind[i]) == 1:
                tr0.append(ls_ind[i])
                tr1.append(ls_rt[i])
                tr2.append(ls_ind[i])
                tr3.append(ls_rt[i][0])
                ls_ind[i] = []
                ls_rt[i] = []
            if len(ls_ind[i]) > 1:
                a = ls_ind[i][ls_rt[i].index(min(ls_rt[i]))]
                for j in range(len(ls_ind[i])):
                   cs = cosine_similarity(S_list[a].reshape((1, 1000)), S_list[ls_ind[i][j]].reshape((1, 1000)))
                   if cs >= 0.95:
                       lst_ind.append(ls_ind[i][j])
                       lst_rt.append(ls_rt[i][j])
                for m in lst_ind:
                     if m in ls_ind[i]:
                         ls_ind[i].remove(m)
                for m in lst_rt:
                     if m in ls_rt[i]:
                         ls_rt[i].remove(m)
                if len(lst_ind)<=len(files):
                    tr0.append(lst_ind)
                    tr1.append(lst_rt)
                    tr2.append(sorted(lst_ind, reverse=True))
                    tr3.append(round(sum(lst_rt)/len(lst_rt),2).astype(np.float32))
                else:
                    while len(lst_ind) > 0:
                        if len(lst_ind) == 1:
                            tr0.append(lst_ind)
                            tr1.append(lst_rt)
                            tr2.append(lst_ind)
                            tr3.append(lst_rt[0])
                            lst_ind = []
                            lst_rt = []
                        if len(lst_ind) > 1:
                            lst1=[]
                            lst2=[]
                            b = lst_ind[lst_rt.index(min(lst_rt))]
                            for j in range(len(lst_ind)):
                               cs = cosine_similarity(S_list[b].reshape((1, 1000)), S_list[lst_ind[j]].reshape((1, 1000)))
                               if cs >= 0.98:
                                   lst1.append(lst_ind[j])
                                   lst2.append(lst_rt[j])
                            tr0.append(lst1)
                            tr1.append(lst2)
                            tr2.append(sorted(lst1, reverse=True))
                            tr3.append(round(sum(lst2)/len(lst2),2).astype(np.float32))
                            for m in lst1:
                               if m in lst_ind:
                                   lst_ind.remove(m)
                            for m in lst2:
                               if m in lst_rt:
                                   lst_rt.remove(m)
                lst_ind = []
                lst_rt = []
    area = []
    areas = []
    for i in range(len(tr0)):
        for j in range(len(tr0[i])):
            area.append(A_list[tr0[i][j]])
        areas.append(area)
        area = []
    rr = []
    rrs = []
    for i in range(len(tr0)):
        for j in range(len(tr0[i])):
            rr.append(R_list[tr0[i][j]])
        rrs.append(rr)
        rr=[]
    areas0 = []
    trs0 = []
    trs1 = []
    rrs0 = []
    for i in range(len(areas)):
        if len(areas[i]) >= 1:
            areas0.append(areas[i])
            rrs0.append(rrs[i])
            trs0.append(tr3[i])
            trs1.append(tr0[i])
    X=np.zeros((len(trs1),len(T)), dtype=int)
    Y=np.zeros((len(trs1),len(T)), dtype=float)
    for i in range(len(trs1)):
        for j in range(len(trs1[i])):
            for t in range(len(T)):
                b = len(T[t])
                if t == 0:
                    a = 0
                if a <= trs1[i][j] < a+b:
                    X[i,t] = areas0[i][j]
                    Y[i,t] = rrs0[i][j]
                a = a+b
    V=np.hstack((X,Y))
    files0=[]
    files1=[]
    for i in range(len(files)):
        files0.append(files[i]+'-'+'area')
        files1.append(files[i]+'-'+'R2')
    files=files0+files1
    max_m = []
    for i in range(len(trs1)):
        max_m.append(np.argmax(S_list[trs1[i][0]])+1)
    df = pd.DataFrame({'rt': trs0, 'max.m/z': max_m})
    for i in range(V.shape[-1]):
       df.insert(loc=len(df.columns), column=files[i], value=V[:,i])

    return df, N
