import numpy as np 
import sys 
import random as rd
import time
import matplotlib.pyplot as plt
from Hammin import *
from BCH import *

Dico = dict([(k,k) for k in range(2)])

def moyenne(L):
    acc, n = 0, len(L)
    for k in range(n):
        acc += L[k]
    return acc/n
    
    
def graph(n):
    T, N = [], []
    for k in range(n):
        Dico = dict([(i,i) for i in range(k)])
        L = []
        for j in range(100):
            start = time.time()
            b = Dico.get(k)
            end = time.time()
            L.append(end - start)
        T.append(moyenne(L))
        N.append(k)
    return T,N
   
def graph2(n):
    T, N = [], []
    for k in range(2, n):
        L1 = [i for i in range(k)]
        L = []
        for j in range(100):
            start = time.time()
            acc = 0 
            while acc < k - 1  and not (L1[acc] == k):
                acc +=1
            b = L1[acc - 2]
            end = time.time()
            L.append(end - start)
        T.append(moyenne(L))
        N.append(k)
    return T,N
    
from random import shuffle

def lookup(x, t):
    for (cle, valeur) in t:
        if x == cle:
            return valeur
    raise KeyError

def creer_erreur(L):
    n = len(L)
    e = [0]*7
    for k in range(n):
        if L[k] >= 0 :
            e[L[k]] = 1 
    return e

Mots_erreur = [creer_erreur(PotH77[k][1]) for k in range(len(PotH77))]
Dicosomme = dict([(str(PotH77[k][0]),Mots_erreur[k]) for k in range(len(PotH77))])    


def table(n):
    k = 2**n
    d = {str(int_to_bin(i)) : i for i in range(k) }
    t = [(str(int_to_bin(i)),i) for i in range(k)]
    L = [str(int_to_bin(i)) for i in range(k)]
    shuffle(L)
    shuffle(t)
    t0 = time.time()
    for cle in L :
        lookup(cle,t)
    print("Liste : %.1f micro-secondes par recherche" % ((time.time() - t0) / k * 1e6))
    t1 = time.time()
    for cle in L:
        d[cle]
    print("Dictionnaire : %.1f micro-secondes par recherche" % ((time.time() - t1) / k * 1e6))
    

X = [2**k for k in range(1,10)]
T1 = [1.5,1.0,0.9,1.9,1.4,2.8,4.9,7.6,18.5]
T2 = [0.6,0.5,0.2,0.4,0.2,0.2,0.3,0.3,0.3]

#plt.plot(X,T1)
#plt.plot(X,T2)
#plt.show()

    
def chrono(n):
    t = [(i, i**2) for i in range(n)]    
    d = {i : i**2 for i in range(n)}
    cles = list(range(n))
    shuffle(cles)
    t0 = time.time()
    for cle in cles:
        lookup(cle, t)
    print("Liste : %.1f micro-secondes par recherche" % ((time.time() - t0) / n * 1e6))
    t1 = time.time()
    for cle in cles:
        d[cle]
    print("Dictionnaire : %.1f micro-secondes par recherche" % ((time.time() - t1) / n * 1e6))



#T, N = graph(1000)
#U,V = graph2(1000)
#plt.plot(V,U)
#plt.plot(N,T)
#plt.show()


def Test(n):
    A = np.array([[[rd.randint(0,255),rd.randint(0,255),rd.randint(0,255)] for k in range(n)] for k in range(n)], dtype="uint8")
    return A

def varie3(T):
    n,m,p = T.shape
    U = np.copy(T)
    N = np.copy(U)
    t0 = time.time()
    for k in range(n):
        for j in range (m):
            for i in range(p):
                fake, truth = envoie3(T[k,j,i])    
                U[k,j,i] = fake%255
                N[k,j,i] = truth
    return time.time() - t0
    
    
def varie_BCH_2(T):
    n,m,p = T.shape
    U = np.copy(T)
    N = np.copy(U)
    t0 = time.time()
    for k in range(n):
        for j in range (m):
            for i in range(p):
                fake, truth = envoie_BCH(T[k,j,i])
                U[k,j,i] = fake%255
                N[k,j,i] = truth
    return  time.time() - t0
    
def trace(k,p):
    X = []
    T = []
    U = []
    for k in range(k,p):
        A = Test(k)
        X.append(k**2)
        T.append(varie3(A))
        U.append(varie_BCH_2(A))
    return X, U, T

##X, U, T = trace(1,100)

#plt.plot(X,T, label = "linéaire")
#plt.plot(X,U, label = "BCH")
#plt.legend()
#plt.show()

def parmi(i,n):
    def fact(k):
        L = [1]
        for i in range(k):
            L.append((i+1)*L[i])
        return L[k]
    return fact(n)/(fact(i)*fact(n-i))

def proba(p,i,n):
    return parmi(i,n)*(p**i)*(1-p)**(n-i)

Z = [proba(0.1,k,7) for k in range(7+1)]
Y = [proba(0.01,k,7) for k in range(7+1)]
X = [proba(0.3,k,7) for k in range(7+1)]
T = [k for k in range(7+1)]

plt.plot(T,X, label = "0.3")
plt.plot(T,Y, label = "0.01")
plt.plot(T,Z, label = "0.1")
plt.legend()
plt.show()
