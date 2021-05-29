import numpy as np

def contract_mod_p(V,p):
    """Contract runs mod p"""
    Vmodp = np.mod(V,p)
    idx = []
    mapping = {}
    min_val_idx = 0
    for i in range(1,len(V)):
        if Vmodp[i]==Vmodp[i-1]:
            if V[i]<V[min_val_idx]:
                min_val_idx = i
        else:
            idx.append(min_val_idx)
            j=i-1
            while Vmodp[i-1] == Vmodp[j] and j>=0:
                mapping[j] = min_val_idx
                j-=1
            min_val_idx = i
    idx.append(min_val_idx)
    j=len(V)-1
    while Vmodp[len(V)-1] == Vmodp[j] and j>=0:
        mapping[j] = min_val_idx
        j-=1
    return V[idx],idx,mapping
def update_S(S_tag,T,idx,mapping,p):
    """Undo the contraction of runs mod p"""
    S = np.zeros(len(mapping))
    S[idx] = S_tag
    for i,min_j in mapping.items():
        S[i] = S[min_j] + (T[i]-T[min_j])/p
    return S

# functions used to implement halving and aliquoting algorithms
def base_dels(t, p):
    return np.ceil(t / p) * p - t
def base_amps(t, p):
    return t - np.floor(t / p) * p
def additional_ops(ti, tprev, si, sprev, p):
    if p * si < ti:
        # amplifications needed
        if p * sprev < tprev:
            # we can reuse some of the amps for tprev
            return np.max((0, (ti - p * si) - (tprev - p * sprev)))
        else:
            return ti - p * si
    elif p * si > ti:
        # deletions needed
        if p * sprev > tprev:
            return np.max((0, (p * si - ti) - (p * sprev - tprev)))
        else:
            return p * si - ti
    else:
        return 0

def odd_runs(V):
    """Compute the number of odd runs; used for halving distance"""
    num_odd_runs = 0
    odd = False
    for t in V:
        if t % 2 == 1:
            if not odd:
                num_odd_runs += 1
                odd = True
        else:
            odd = False
    return num_odd_runs

def cnd_halving(T):
    """Halving algorithm"""
    S = np.ceil(T / 2.0)
    return odd_runs(T), S

def reconstruct_indexes(C, m):
    """Compute an optimal preduplication profile from DP tables"""
    n = C.shape[0]
    a = C[n-1].argmin()
    x = m[n-1, a]
    i = n-1
    while i >= 2:
        i -= 1
        x = m[i, int(x)]
        yield int(x)

def cnd_aliquoting_I(T, p=2):
    """Implementation of aliquoting using O(n^2) time.
    Does not handle zero coords properly; preprocess to remove them."""
    n = len(T)
    C = np.empty((n, 2*n))
    C[:] = np.inf
    m = np.empty((n, 2*n), dtype=int)
    m[:] = -1
    for i in range(n):
        ti = T[i]

        for x in range(2*n):
            # if x < n, then we are doing base + x*p amplifications.
            # if x >= n, then we are doing base + (x - n)*p deletions.
            idx = x
            o = x if x < n else (x - n)
            # x uniquely determines si for our kind of solution
            si = (np.floor(ti / p) - o) if x < n else (np.ceil(ti / p) + o)
            if si <= 0:
                C[i, idx] = np.inf
            else:
                if i == 0:
                    C[i, idx] = np.abs(p*si - ti)
                else:

                    argmin = None
                    val = np.inf

                    if x >= n: # n - x deletions
                        ops = base_dels(T[i], p) + o * p
                        prev_base_ops = base_dels(T[i-1], p)
                        y1_idx = n + int(np.ceil((ops - prev_base_ops) / p))
                        y2_idx = n + int(np.floor((ops - prev_base_ops) / p))

                    else:
                        ops = base_amps(T[i], p) + o * p
                        prev_base_ops = base_amps(T[i-1], p)
                        y1_idx = int(np.ceil((ops - prev_base_ops) / p))
                        if T[i-1] - prev_base_ops - y1_idx * p < p:
                            y1_idx = 0
                        y2_idx = int(np.floor((ops - prev_base_ops) / p))
                        y2_idx = min(int(np.floor((T[i-1]) / p))-1, y2_idx)

                    for y in [y1_idx, y2_idx, 0, n]:
                        if (y < 0) or (y >= 2*n): continue
                        if y < n:
                            additional = additional_ops(ti, T[i-1], si, np.floor(T[i-1] / p) - y, p)
                        else:
                            additional = additional_ops(ti, T[i-1], si, np.ceil(T[i-1] / p) + (y - n), p)
                        ops = C[i-1, y] + additional
                        if ops < val:
                            val = ops
                            argmin = y

                    C[i, idx] = val
                    m[i, idx] = argmin


    # reconstruct preduplication profile
    a = C[n-1].argmin()
    x = m[n-1, a]
    l = reversed([a, int(x)] + list(reconstruct_indexes(C, m)))
    S = list(map(lambda x: (np.floor(x[0] / p) - x[1]) if x[1] < n
                 else (np.ceil(x[0] / p) + (x[1] - n)), zip(T,l)))
    S = np.array(S)

    return C[n-1].min(), S, C, m


def cnd_aliquoting_dp(T, p=2, plot=False):
    """Implementation of aliquoting using O(n^3) time, and optionally
    generates a figure to visualize an optimal aliquoting CNT.
    Does not handle zero coords properly; exclude them."""
    n = len(T)
    C = np.empty((n, 2*n))
    C[:] = np.inf
    m = np.empty((n, 2*n))
    m[:] = np.nan
    for i in range(n):
        ti = T[i]

        for x in range(2*n):
            # if x < n, then we are doing x amplifications.
            # if x >= n, then we are doing x - n deletions.
            idx = x
            o = x if x < n else (x - n)
            # x uniquely determines si for our kind of solution
            si = (np.floor(ti / p) - o) if x < n else (np.ceil(ti / p) + o)
            if si <= 0:
                C[i, idx] = np.inf
            else:
                if i == 0:
                    C[i, idx] = np.abs(p*si - ti)
                else:
                    argmin = None
                    val = np.inf
                    for y in range(n):
                        ops = C[i-1, y] + additional_ops(ti, T[i-1], si, np.floor(T[i-1] / p) - y, p)
                        if ops < val:
                            val = ops
                            argmin = y
                    for y in range(n):
                        ops = C[i-1, n+y] + additional_ops(ti, T[i-1], si, np.ceil(T[i-1] / p) + y, p)
                        if ops < val:
                            val = ops
                            argmin = y+n
                    C[i, idx] = val
                    m[i, idx] = argmin

    # reconstruct preduplication profile
    a = C[n-1].argmin()
    x = m[n-1, a]
    l = reversed([a, int(x)] + list(reconstruct_indexes(C, m)))
    S = list(map(lambda x: (np.floor(x[0] / p) - x[1]) if x[1] < n
                 else (np.ceil(x[0] / p) + (x[1] - n)), zip(T,l)))
    S = np.array(S)
    pS = p * S
    if plot:
        from matplotlib import pyplot as plt
        ax = plt.figure().gca()
        ax.yaxis.get_major_locator().set_params(integer=True)
        plt.bar(range(n), (T - pS))
        plt.grid(True, axis='y')
        plt.show()
        plt.close()
    return C[n-1].min(), S, C, m
