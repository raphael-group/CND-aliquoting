### imports
import numpy as np
import scipy
import sys
import os
#import multiprocessing as mp
import pathos.multiprocessing as mp
import pandas as pd
from gurobipy import *

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

def DirectedOrderedLp(u,v,test=False,debug=False):
    if not validateProfiles(u,v):
        return np.inf
    u,v,n,M = directed_cnd_prefix(u,v)
    model = Model("DirectedOrderedLp")
    # Create variables
    d,t = {},{}
    signs = ['- before','+','- after']
    for s in signs:
        d[s,i] = model.addVar(lb=0, ub=M, obj=0)
        t[s,i] = model.addVar(lb=0, ub=M, obj=0)
    d['-',-1] = d['+',-1] = 0
    model.update()
    #add constraints
    for i in range(n):
        if v[i]==0:
            model.addConstr(u[i],'<=',d['-',i],'u['+str(i)+']<=d[-,'+str(i)+']')
        else:
            model.addConstr(u[i]-d['-',i]+d['+',i]==v[i],'v['+str(i)+'] equality')
            model.addConstr(d['-',i]<=u[i]-1,'m['+str(i)+']-1>=d[v,-,'+str(i)+']')
    for i in range(n):
        for s in signs:
            model.addConstr(t[s,i]>=d[s,i]-d[s,i-1]) 
    model.update()
    model.setParam('OutputFlag', False )
    # Solve the LP
    model.optimize()
    if model.status == GRB.status.INFEASIBLE:
        print(model.ModelName,"Infeasible!!!")
        return np.inf
    if debug:
        for s in signs:
            print_events_summaries(np.array([d[s,i].x for i in range(n)]),s)
        print("total",model.objVal)
    return model.objVal

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

EuclideanDistance=lambda x,y: np.sqrt(np.sum((np.array(x)-np.array(y))**2))
LOneDistance=lambda x,y: np.sum(np.abs(np.array(x)-np.array(y)))

## Weighted functions

def DirectedSemiOrderedLp(u,v,test=False,debug=False,match_ops=False,weight_op=lambda s,i,j,n: 1):
    if not validateProfiles(u,v):
        return np.inf
    u,v,n,M = directed_cnd_prefix(u,v)
    d,t = {},{}
    signs = ['- before','+','- after']
    for i in range(n):
        for s in signs:
            d[s,i] = model.addVar(lb=0, ub=M, obj=0)
            t[s,i] = model.addVar(lb=0, ub=M, obj=0)
    for s in signs:
        d[s,-1] = 0
    if match_ops:
        for s in signs:
            d[s,n] = 0
        x,q={},{}
        for i in range(n):
            for j in range(i+1,n+1):
                for s in signs:
                    x[s,i,j] = model.addVar(lb=0, ub=M, obj=0)
        for j in range(1,n+1):
            for s in signs:
                q[s,j] = model.addVar(lb=0, ub=M, obj=0)
    model.setObjective(quicksum(t[s,i] for i in range(n) for s in signs),GRB.MINIMIZE) 
    model.update()
    for i in range(n):
        if v[i]==0:
            model.addConstr(u[i],'<=',d['- before',i])
        else:
            model.addConstr(u[i]-d['- before',i]+d['+',i]-d['- after',i]==v[i])
            model.addConstr(d['- before',i]<=u[i]-1)
        for s in signs:
            model.addConstr(t[s,i]>=d[s,i]-d[s,i-1])
    if match_ops:
        for j in range(1,n+1):
            for s in signs:
                model.addConstr(q[s,j]>=d[s,j-1]-d[s,j])
        for i in range(n):
            for s in signs:
                model.addConstr(quicksum(x[s,i,j] for j in range(i+1,n+1))==t[s,i])
        for j in range(1,n+1):
            for s in signs:
                model.addConstr(quicksum(x[s,i,j] for i in range(j))==q[s,j])        
        model.setObjectiveN(quicksum(weight_op(s,i,j,n)*x[s,i,j] for i in range(n) for j in range(i+1,n+1) for s in signs),1,0) 
    model.update()
    model.setParam('OutputFlag', False )
    # Solve the LP
    model.optimize()
    if model.status == GRB.status.INFEASIBLE:
        print(model.ModelName,"Infeasible!!!")
        return np.inf
    if debug:
        for s in signs:
            if not match_ops:
                print_events_summaries(np.array([d[s,i].x for i in range(n)]),s)
            else:
                print('d',s,[(i,d[s,i].x) for i in range(n) if d[s,i].x>0])
                print('t',s,[(i,t[s,i].x) for i in range(n) if t[s,i].x>0])
                print('q',s,[(j,q[s,j].x) for j in range(1,n+1) if q[s,j].x>0])
                print('x',s,[(i,j,x[s,i,j].x) for i in range(n) for j in range(i+1,n+1) if x[s,i,j].x>0])
    if match_ops:
        model.setParam(GRB.Param.ObjNumber, 1)
        return model.ObjNVal,{s:[(i,j,x[s,i,j].x) for i in range(n) for j in range(i+1,n+1) if x[s,i,j].x>0] for s in signs}
    return model.objVal

def op_to_subkind(i,j,n):
    if i==0 and j==n:
        return 'whole'
    elif j-i==1:
        return 'small'
    elif i==0 or j==n:
        return 'arm'
    else:
        return 'segmental'
    
def weigh_ops_with_prob(stat):
    return lambda s,i,j,n: -np.log(stat.loc[s,op_to_subkind(i,j,n)])

def cn_breakpoints(vec):
    return [0]+list(np.where(vec[1:] != vec[:-1])[0]+1)+[len(vec)]

def WeightedDistance(u,v,debug=False,weight_op=lambda s,i,j,n: 1,wgd=0,min_ops=False):
    if not validateProfiles(u,v):
        return np.inf,{}
    #u,v,n,M = directed_cnd_prefix(u,v)
    u,v,n,M = np.array(u),np.array(v),len(u),max(max(v),max(u))
    if (u==v).all():
        return 0,{}
    model = Model("WeightedDistance")
    d,x = {},{}
    signs = ['- before','+','- after']
    for i in range(n):
        for s in signs:
            d[s,i] = model.addVar(lb=0, ub=M, obj=0)
    breakpoints = sorted(list(set(cn_breakpoints(u)).union(cn_breakpoints(v))))
    pairs = [(i,j) for ind,i in enumerate(breakpoints[:-1]) for j in breakpoints[ind+1:]]
    for i,j in pairs:
        for s in signs:
            x[s,i,j] = model.addVar(lb=0, ub=M, obj=0)            
    model.setObjective(quicksum(weight_op(s,i,j,n)*x[s,i,j] for i,j in pairs for s in signs),GRB.MINIMIZE)
    model.update()
    for k in range(n):
        for s in signs:
            model.addConstr(d[s,k]==quicksum(x[s,i,j] for i,j in pairs if i<=k and k<j))
    for i in range(n):
        if v[i]==0:
            model.addConstr(u[i],'<=',d['- before',i])
        else:
            model.addConstr(u[i]-d['- before',i]+d['+',i]-d['- after',i]==v[i])
            model.addConstr(d['- before',i]<=u[i]-1)
    if wgd>0:
        model.addConstr(x['+',0,n]>=wgd)
    if min_ops:
        min_len = DirectedCopyNumberDistanceLinear(u,v)
        model.addConstr(quicksum(x[s,i,j] for i,j in pairs for s in signs)<=min_len)
    model.update()
    model.setParam('OutputFlag', False )
    model.optimize()
    if model.status == GRB.status.INFEASIBLE:
        print(model.ModelName,"Infeasible!!!")
        return np.inf
    if debug:
        for s in signs:
            print('d',s,[(i,d[s,i].x) for i in range(n) if d[s,i].x>0])
            print('x',s,[(i,j,x[s,i,j].x) for i,j in pairs if x[s,i,j].x>0])
    return model.objVal,{s:[(i,j,x[s,i,j].x) for i,j in pairs if x[s,i,j].x>0] for s in signs}

def WeightedDistance_pratial(args):
    return WeightedDistance(args[0],args[1],weight_op=args[2],wgd=args[3],min_ops=args[4])



def CopyNumberDistanceSymmetric(u,v,test=False,debug=False,guess=None):
    u,v,n,B = directed_cnd_prefix(u,v)
    if n==0:
        return 0
    if validateProfiles(u,v):
        return DirectedCopyNumberDistanceLinear(u,v)
    elif validateProfiles(v,u):
        return DirectedCopyNumberDistanceLinear(v,u)
    model = Model("cnp_median")
    # Create variables
    d,t,M = {},{},{}
    Y = {'u':u,'v':v}
    profiles = ['u','v']
    signs = ['-','+']
    for i in range(n):
        M[i] = model.addVar(vtype=GRB.INTEGER,lb=1, ub=B, obj=0,name='m['+str(i)+']')
        for p in profiles:
            for s in signs:
                d[i,p,s] = model.addVar(lb=0, ub=B, obj=0,name='d['+str(i)+','+str(p)+','+s+']')
                t[i,p,s] = model.addVar(lb=0, ub=B, obj=1,name='t['+str(i)+','+str(p)+','+s+']')    
    for p in profiles:
        for s in signs:
            d[-1,p,s] = 0
    model.update()
    for i in range(n):
        for p in profiles:
            if Y[p][i]==0:
                model.addConstr(M[i]<=d[i,p,'-'])
            else:
                model.addConstr(M[i]-d[i,p,'-']+d[i,p,'+']==Y[p][i])
                model.addConstr(d[i,p,'-',]<=M[i]-1)
    for i in range(n):
        for p in profiles:
            for s in signs:
                model.addConstr(t[i,p,s]>=d[i,p,s]-d[i-1,p,s])   
    if not guess is None:
        for i in range(n):
            model.addConstr(M[i]==guess[i])
    model.update()
    model.setParam('OutputFlag', False )
    model.setParam(GRB.Param.MIPGapAbs, 0.9)
    # Solve the LP
    model.optimize()
    if model.status == GRB.status.INFEASIBLE:
        print(model.ModelName,"Infeasible!!!")
        return -1
    if debug:
        print('M=',[M[i].x for i in range(n)])
    return model.objVal

def SymmetricWeightedDistance(u,v,debug=False,weight_op=lambda s,i,j,n: 1):
    u,v,n,B = directed_cnd_prefix(u,v)
    if (u==v).all():
        return 0
    model = Model("SymmetricWeightedDistance")
    d,x,M = {},{},{}
    Y = {'u':u,'v':v}
    profiles = ['u','v']
    signs = ['- before','+','- after']
    for i in range(n):
        M[i] = model.addVar(vtype=GRB.INTEGER,lb=1, ub=B, obj=0,name='m['+str(i)+']')
        for s in signs:
            for p in profiles:
                d[p,s,i] = model.addVar(lb=0, ub=B, obj=0)
    breakpoints = sorted(list(set(cn_breakpoints(u)).union(cn_breakpoints(v))))
    pairs = [(i,j) for ind,i in enumerate(breakpoints[:-1]) for j in breakpoints[ind+1:]]
    for i,j in pairs:
        for s in signs:
            for p in profiles:
                x[p,s,i,j] = model.addVar(lb=0, ub=B, obj=0)            
    model.setObjective(quicksum(weight_op(s,i,j,n)*x[p,s,i,j] for i,j in pairs for s in signs for p in profiles),GRB.MINIMIZE)
    model.update()
    for k in range(n):
        for s in signs:
            for p in profiles:
                model.addConstr(d[p,s,k]==quicksum(x[p,s,i,j] for i,j in pairs if i<=k and k<j))
    for i in range(n):
        for p in profiles:
            if Y[p][i]==0:
                model.addConstr(M[i]<=d[p,'- before',i])
            else:
                model.addConstr(M[i]-d[p,'- before',i]+d[p,'+',i]-d[p,'- after',i]==Y[p][i])
                model.addConstr(d[p,'- before',i]<=M[i]-1)
    model.update()
    model.setParam('OutputFlag', False )
    model.optimize()
    if model.status == GRB.status.INFEASIBLE:
        print(model.ModelName,"Infeasible!!!")
        return np.inf
    if debug:
        print('M=',[M[i].x for i in range(n)])
    return model.objVal

# def dummy_weight(s,i,j,n):
#     if i==0 and j==n:
#         return 1.3
#     elif j-i==1:
#         return 0.9
#     elif i==0 or j==n:
#         return 1.6
#     else:
#         return 1.1

def semi_directed_cnd(u,v,dist_func=DirectedCopyNumberDistanceLinear):
    if validateProfiles(u,v):
        return dist_func(u,v)
    u_new,v_new=np.copy(u),np.copy(v)
    x = np.where(np.logical_and(u_new==0,v_new>0))[0]
    u_new[x]=1
    dummy_v=np.ones(len(v_new))
    dummy_v[x]=0
    additional_d = dist_func(np.ones(len(u_new)),dummy_v)
    d = dist_func(u_new,v_new)
    return d+additional_d

def semi_symmetrized_cnd(u,v,dist_func=DirectedCopyNumberDistanceLinear):
    return (semi_directed_cnd(u,v,dist_func)+semi_directed_cnd(v,u,dist_func))/2
