DATASET_ORIGINAL = "web-Stanford.txt" #Uncompress the .gz file!
NODES = 281903
EDGES = 2312497
# DATASET_ORIGINAL='edges.txt'
# NODES=4
# EDGES=8
beta=0.85
epsilon=10**-4
import numpy as np
from  scipy import sparse

def dataset2csr():
    row = []
    col = []    
    with open(DATASET_ORIGINAL,'r') as f:
        for line in f.readlines()[4:]:
            origin, destiny = (int(x)-1 for x in line.split())
            row.append(destiny)
            col.append(origin)
    return(sparse.csr_matrix(([1]*EDGES,(row,col)),shape=(NODES,NODES)))
    #return(sparse.csr_matrix(([True]*EDGES,(row,col)),shape=(NODES,NODES)))

csr_m = dataset2csr()

import sys

print "The size in memory of the adjacency matrix is {0} MB".format(
    (sys.getsizeof(csr_m.shape)+
    csr_m.data.nbytes+
    csr_m.indices.nbytes+
    csr_m.indptr.nbytes)/(1024.0**2)
)

# def csr_save(filename,csr):
#     np.savez(filename,
#         nodes=csr.shape[0],
#         edges=csr.data.size,
#         indices=csr.indices,
#         indptr =csr.indptr
#     )

# def csr_load(filename):
#     loader = np.load(filename)
#     edges = int(loader['edges'])
#     nodes = int(loader['nodes'])
#     return sparse.csr_matrix(
#         (np.bool_(np.ones(edges)), loader['indices'], loader['indptr']),
#         shape = (nodes,nodes)
#     )

# DATASET_NATIVE = 'dataset-native.npz'
# csr_save(DATASET_NATIVE,csr_m)
# csr = csr_load(DATASET_NATIVE)

csr=csr_m
csr.data.tofile('edata.txt',sep=' \n')
csr.indptr.tofile('eindptr.txt',sep=' \n')
csr.indices.tofile('eindices.txt',sep=' \n')
deg_out_beta=csr.sum(axis=0).T
deg_out_beta.tofile('eout.txt',sep=' \n')

def compute_PageRank(G):
    '''
    Efficient computation of the PageRank values using a sparse adjacency 
    matrix and the iterative power method.
    
    Parameters
    ----------
    G : boolean adjacency matrix. np.bool8
        If the element j,i is True, means that there is a link from i to j.
    beta: 1-teleportation probability.
    epsilon: stop condition. Minimum allowed amount of change in the PageRanks
        between iterations.

    Returns
    -------
    output : tuple
        PageRank array normalized top one.
        Number of iterations.

    '''    
    #Test adjacency matrix is OK
    n,_ = G.shape
    assert(G.shape==(n,n))
    #Constants Speed-UP
    deg_out_beta = G.sum(axis=0).T/beta #vector
    #Initialize
    ranks = np.ones((n,1))/n #vector
    time = 0
    flag = True
    while flag:        
        time +=1
        with np.errstate(divide='ignore'): # Ignore division by 0 on ranks/deg_out_beta
            new_ranks = G.dot((ranks/deg_out_beta)) #vector
        #Leaked PageRank
        new_ranks += (1-new_ranks.sum())/n
        #Stop condition
        if np.linalg.norm(ranks-new_ranks,ord=1)<=epsilon:
            flag = False        
        ranks = new_ranks
    return(ranks, time)

print '==> Computing PageRank'
pr,iters = compute_PageRank(csr)
print pr
print '\nIterations: {0}'.format(iters)
print 'Element with the highest PageRank: {0}'.format(np.argmax(pr)+1)



# while flag:
#     time++
#     tmp=ranks/deg_out_beta
#     new_ranks=G*tmp
#     sum=new_ranks.sum()
#     sum=(1-sum)/n
#     new_ranks=new_ranks+sum
#     tmp=ranks-new_ranks
#     sum=tmp.sum_abs()
#     if(sum<=eps):
#         flag=False
#     ranks=new_ranks