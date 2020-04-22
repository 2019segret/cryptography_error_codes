import numpy as np 
import sys 
import random as rd
import time
from mat import *


Table =[[(-1,0),(1,3),(1,6),(1,1),(1,5),(1,4),(1,2)],
        [(1,3),(-1,0),(1,4),(1,0),(1,2),(1,6),(1,5)],
        [(1,6),(1,4),(-1,0),(1,5),(1,1),(1,3),(1,0)],
        [(1,1),(1,0),(1,5),(-1,0),(1,6),(1,2),(1,4)],
        [(1,5),(1,2),(1,1),(1,6),(-1,0),(1,0),(1,3)],
        [(1,4),(1,6),(1,3),(1,2),(1,0),(-1,0),(1,1)],
        [(1,2),(1,5),(1,0),(1,4),(1,3),(1,1),(-1,0)]]


def alpha_somme(x,y) :
    return Table[x[1]][y[1]]


def modulo(L):
    for k in range(len(L)):
        L[k] = L[k]%2
    return L
    
    
g = [1,1,0,1,0,0,0]



def produit_polynome(P,Q):
    d_p=len(P)-1 
    d_q=len(Q)-1  
    PQ=[0 for i in range(d_p + d_q + 1)] # nouv polynôme de d° = d°p x d°q
    for i in range(d_p+d_q+1):
        coeff_PQ_deg_i=0 #coeff_deg_i pour calculer le coefficient (PQ)_i 
        for j in range(i+1):                          
            if j<=d_p : coeff_P_deg_j=P[j]
            else:coeff_P_deg_j=0             
            if (i-j)<=d_q : coeff_Q_deg_j=Q[i-j]
            else: coeff_Q_deg_j=0             
            coeff_PQ_deg_i += coeff_Q_deg_j*coeff_P_deg_j
        PQ[i]=coeff_PQ_deg_i
    return PQ[:7]


def code(P):
    a = modulo(produit_polynome(g,P))
    return a
    
def est_nulle(P):
    return P == [0]*len(P)


def evalue(P):
    if est_nulle(P):
        return [(-1,0)]
    else : 
        L = []
        for k in range(len(P)):
            if P[k] == 1 :
                L.append((1,k))
    return L 

def simpl(L):
    L[0] = alpha_somme(L[0],L[1])
    del L[1]
    return L

def simplifie(L):
    acc = L[0]
    for k in range (1,len(L)):
        if acc != (-1,0) :
            acc = alpha_somme(acc,L[k])
        else : 
            acc = L[k]
    return acc

###########################

    
def decode (P):
    A = simplifie(evalue(P))
    if A[0] == -1 :
        return P
    else : 
        P[A[1]] = (1 + P[A[1]])%2 
    return P
    
def aleatoire(t):
    n = len(t)
    a = rd.randint(0,n-1)
    t[a] = (t[a] + 1)%2 
    return t
    
def int_to_bin2(n):
    l = []
    q = - 1 
    r = ''
    while n > 0 :
        r = n%2
        l.append(int(r))
        n = (n-r)/2
    k = len(l) 
    l.reverse()
    return [0 for i in range(8-k)] + l
    
def switch_BCH(k):
    a = int_to_bin2(k)
    U,V = code(a[:4]), code(a[4:])
    return aleatoire(U), aleatoire(V)


Dictio2 = dict([(str(int(bin_to_int(code(Mots77[k])))),Mots77[k]) for k in range(len(Mots77)) ])

def associe(A):
    A.reverse()
    a = Dictio2.get(str(int(bin_to_int(A))))
    return a

def final_BCH(u,v):
    a, b = associe(decode(u)), associe(decode(v))
    return int(bin_to_int(module1(np.concatenate((a,b)))))

def bin_to_int(t):
    n = len(t)
    acc = 0
    for k in range(n):
        acc += t[k]*(2**k) #n-k-1
    return acc


#########################################

def code_divise(k):
    a = int_to_bin2(k)
    U,V = code(a[:4]), code(a[4:])
    return U,V


def envoie_BCH(k):
    if rd.random() < 0.1 :
        s,v = switch_BCH(k)
        a, b = str(int(bin_to_int(s))), str(int(bin_to_int(v)))
        return int(a+b),final_BCH(s,v)
    else :
        s,v = code_divise(k)
        a = final_BCH(s,v)
        return a, a

    
Gandalf = Pixels('gandalf.jpg')
    
            
def varie_BCH(T):
    n,m,p = T.shape
    U = np.copy(T)
    N = np.copy(U)
    for k in range(n):
        for j in range (m):
            for i in range(p):
                fake, truth = envoie_BCH(T[k,j,i])
                U[k,j,i] = fake%255
                N[k,j,i] = truth
    return U, N 
 
#start_time = time.time()
    
#Fake, Truth = varie_BCH(Gandalf)

#print("Temps d execution : %s secondes ---" % (time.time() - start_time))