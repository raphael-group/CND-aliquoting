### imports
import numpy as np
import scipy
import sys
import os

### CND functions

def compress_cn_mat(mat):
    n,m = mat.shape
    y = mat[:,1:] != mat[:,:m-1]
    x = np.any(y,axis=0)
    x = np.hstack((True,x))
    return mat[:,x]

def validateProfiles(u,v):
    return not np.logical_and(np.array(u)==0,np.array(v)>0).any()
                
def print_events_summaries(e,name):
    x = np.clip(np.hstack((e[0],e[1:]-e[:-1],0)),0,np.inf)
    y = np.clip(np.hstack((0,e[:-1]-e[1:],e[-1])),0,np.inf)
    print(name+' starts',list(zip(list(np.where(x>0)[0]),list(x[x>0]))))
    print(name+' ends',list(zip(list(np.where(y>0)[0]),list(y[y>0]))))

def directed_cnd_prefix(u,v):
    if len(v)!=len(u):
        raise ValueError("Length doesn't match")
    u,v=np.array(u),np.array(v)
    u,v=u[np.logical_not(np.logical_and(u==0,v==0))],v[np.logical_not(np.logical_and(u==0,v==0))]
    n = len(u)
    if n==0:
        return u,v,n,0
    M = max(max(v),max(u))
    return u,v,n,M

def calc_Q(u,v):
    n=len(u)
    u_m = 0
    prev = -1
    Q = {}
    for i in range(n):
        if v[i]==0:
            u_m = max(u_m,u[i])
        else:
            Q[i] = (u_m,prev)
            prev = i
            u_m = 0
    Q[n] = (u_m,prev)
    return Q

def CalcPmin(ui,vi):
    return max(ui-vi,0)

def CalcPmax(ui):
    return max(ui-1,0)

def CalcM(ui,vi,ui1,vi1):
    return (vi-ui)-(vi1-ui1)

def limit(x,down,up):
    if x<down:
        return down
    if x>up:
        return up
    return x

class MyFunc:
    """This is my function"""
    def __init__(self,ui,vi,a,b,base):
        self.ui = ui
        self.vi = vi
        self.pmin = CalcPmin(ui,vi)
        self.pmax = CalcPmax(ui)
        self.a = a
        self.b = b
        self.base = base
    def CalcP(self,p):
        if p<self.pmin or p>self.pmax:
            raise("Error - unvalid p")
        if p<=self.a:
            return self.base
        if p<=self.b:
            return self.base + p - self.a
        if p<=self.pmax:
            return self.base - self.b - self.a + 2*p
    def CalcNext(self,ui,vi,Qi):
        Mi = CalcM(ui,vi,self.ui,self.vi)
        pmin = CalcPmin(ui,vi)
        pmax = CalcPmax(ui)
        if Mi>=0:
            if Qi<=self.a:
                nextBase = self.base
                nextA = self.a-Mi
                nextB = self.b
            elif Qi<=self.b:
                nextBase = self.base + Qi - self.a
                nextA = Qi-Mi
                nextB = self.b
            elif Qi>=self.b:
                nextBase = self.base + Qi - self.a
                nextA = self.b-Mi
                nextB = Qi
        elif Mi<=0:
            if Qi<=self.a:
                nextBase = self.base
                nextA = self.a
                nextB = self.b-Mi
            elif Qi<=self.b:
                nextBase = self.base + Qi - self.a
                nextA = Qi
                nextB = self.b-Mi
            elif Qi>=self.b:
                nextBase = self.base + Qi - self.a
                nextA = min(self.b-Mi,Qi)
                nextB = max(Qi,self.b-Mi)
        if pmin > nextA and pmin<=nextB:
            nextBase = nextBase + pmin - nextA
        if pmin > nextB:
            nextBase = nextBase - nextB - nextA + 2*pmin
        nextA = limit(nextA,pmin,pmax)
        nextB = limit(nextB,nextA,pmax)
        return MyFunc(ui,vi,nextA,nextB,nextBase)

def DirectedCopyNumberDistanceLinear(u,v):
    u,v,n,M = directed_cnd_prefix(u,v)
    if len(v)!=len(u):
        raise ValueError("Length doesn't match")
    if n==0 or (u==v).all():
        return 0
    else:
        u,v=compress_cn_mat(np.array([u,v]))
        n=len(u)
    Q = calc_Q(u,v)
    prevFunc = MyFunc(M+1,M+1,0,0,0)
    for i in range(n):
        if v[i]>0:
            prevFunc = prevFunc.CalcNext(u[i],v[i],Q[i][0])
    u_m,prev = Q[n]
    if prev<0:
        return u_m
    p_min = CalcPmin(u[prev],v[prev])
    d = prevFunc.CalcP(p_min)+max(u_m-p_min,0)
    for p in range(int(p_min)+1,int(prevFunc.pmax)+1):
        d = min(d,prevFunc.CalcP(p)+max(u_m-p,0))
    return d
