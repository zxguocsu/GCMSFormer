# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 10:50:38 2023

@author: fanyingjie
"""
import numpy as np
import seaborn
seaborn.set_context(context="talk")
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from GCMSFormer.Resolution import Resolution, output_msp
import warnings
warnings.filterwarnings("ignore")

def Alignment(path,files, model,tgt_vacob,device):
    T = []
    S = []
    A = []
    R = []
    N = []
    for filename in files:
        sta_S, area, rt, R2 = Resolution(path,filename,model,tgt_vacob,device)
        msp = filename.split('.CDF')[0] + '.MSP'
        output_msp(path + '/'+ msp, sta_S, rt)
        T.append(rt)
        S.append(sta_S)
        A.append(area)
        R.append(R2)
        N.append((np.vstack((np.array(rt),np.array(area),np.array(R2)))).T)
        T_list = np.array(T[0])
        S_list = np.array(S[0])
        A_list = np.array(A[0])
        R_list = np.array(R[0])
        for i in range(1, len(S)):
            S_list = np.vstack((S_list, np.array(S[i])))
            T_list = np.hstack((T_list, np.array(T[i])))
            A_list = np.hstack((A_list, np.array(A[i])))  
            R_list = np.hstack((R_list, np.array(R[i]))) 
    index_rt = np.argsort(T_list)
    index_list = np.sort(T_list)
    l  = [index_rt[0]] 
    ls = []
    for i in range(1, len(index_list)):
        if index_list[i]-index_list[i-1] <= 0.1:
            l.append(index_rt[i])
        else:
            ls.append(l)
            l = [index_rt[i]]
            if i == len(index_list)-1:
                ls.append(l)
    l = [index_list[0]] 
    ls0 = []
    for i in range(1, len(index_list)):
        if index_list[i]-index_list[i-1] <= 0.1:
            l.append(index_list[i])
        else:
            ls0.append(l)
            l = [index_list[i]]
            if i == len(index_list)-1:
                ls0.append(l)        
    lst = []
    lst0 = []
    tr0 = []
    tr1 = []
    tr2 = []
    tr3 = []
    for i in range(len(ls)):
        while len(ls[i]) > 0:
            if len(ls[i]) == 1:
                tr0.append(ls[i])
                tr1.append(ls0[i])
                tr2.append(ls[i])
                tr3.append(ls0[i][0])
                ls[i] = []
                ls0[i] = []
            if len(ls[i]) > 1:
                a = ls[i][ls0[i].index(min(ls0[i]))]
                for j in range(len(ls[i])):
                   cs = cosine_similarity(S_list[a].reshape((1, 1000)), S_list[ls[i][j]].reshape((1, 1000)))
                   if cs >= 0.95:
                       lst.append(ls[i][j])
                       lst0.append(ls0[i][j])
                for m in lst: 
                     if m in ls[i]: 
                         ls[i].remove(m)
                for m in lst0: 
                     if m in ls0[i]: 
                            ls0[i].remove(m)
                if len(lst)<=len(files):
                    tr0.append(lst)
                    tr1.append(lst0)
                    tr2.append(sorted(lst, reverse=True))
                    tr3.append(round(sum(lst0)/len(lst0),2).astype(np.float32))
                else:
                    while len(lst) > 0:
                        if len(lst) == 1:
                            tr0.append(lst)
                            tr1.append(lst0)
                            tr2.append(lst)
                            tr3.append(lst0[0])
                            lst = []
                            lst0 = []
                        if len(lst) > 1:
                            lst1=[]
                            lst2=[]
                            b = lst[lst0.index(min(lst0))]
                            for j in range(len(lst)):
                               cs = cosine_similarity(S_list[b].reshape((1, 1000)), S_list[lst[j]].reshape((1, 1000)))
                               if cs >= 0.98:
                                   lst1.append(lst[j])
                                   lst2.append(lst0[j])
                            tr0.append(lst1)
                            tr1.append(lst2)
                            tr2.append(sorted(lst1, reverse=True))
                            tr3.append(round(sum(lst2)/len(lst2),2).astype(np.float32))
                            for m in lst1: 
                               if m in lst: 
                                   lst.remove(m)
                            for m in lst2: 
                               if m in lst0: 
                                   lst0.remove(m)                        
                lst = []
                lst0 = [] 
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
        if len(areas[i])>3:
            areas0.append(areas[i])
            rrs0.append(rrs[i])
            trs0.append(tr3[i])
            trs1.append(tr0[i])            
    X=np.zeros((len(trs1),len(T)), dtype=int)
    Y=np.zeros((len(trs1),len(T)), dtype=float)
    for i in range(len(trs1)):
        for j in range(len(trs1[i])):
            for t in range(len(T)):
                b=len(T[t])
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