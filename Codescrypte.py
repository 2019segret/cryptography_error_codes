``` python

import numpy as np 
import sys 
import random as rd
from numpy.polynomial import Polynomial

Table =[[(-1,0),(1,3),(1,6),(1,1),(1,5),(1,4),(1,2)],
        [(1,3),(-1,0),(1,4),(1,0),(1,2),(1,6),(1,5)],
        [(1,6),(1,4),(-1,0),(1,5),(1,1),(1,3),(1,0)],
        [(1,1),(1,0),(1,5),(-1,0),(1,6),(1,2),(1,4)],
        [(1,5),(1,2),(1,1),(1,6),(-1,0),(1,0),(1,3)],
        [(1,4),(1,6),(1,3),(1,2),(1,0),(-1,0),(1,1)],
        [(1,2),(1,5),(1,0),(1,4),(1,3),(1,1),(-1,0)]]


Mots = np.array([[0,1,0,0,1,0,1,1,0,1,0,1,0,1,0,1],
                 [0,0,1,1,1,0,0,1,0,0,1,1,0,0,1,1],
                 [0,0,1,0,0,1,1,1,0,0,0,0,1,1,1,1],
                 [0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1],      
                 [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                 [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                 [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]])

Mots77 = np.transpose(Mots)


def alpha_somme(x,y) :
    return Table[x[1]][y[1]]

g = [1,1,0,1,0,0,0]

def modulo (L):
    for k in range(len(L)):
        L[k] = L[k] % 2 
    return L

def produit_polynome(P,Q):
    d_p=len(P)-1 # le degré de P
    d_q=len(Q)-1 # le degré de Q     
    PQ=[0 for i in range(d_p + d_q+1)] # création de la polynôme PQ=PxQ de degré d_pq = d_p + d_q     
    for i in range(d_p+d_q+1):
        coeff_PQ_deg_i=0 #coeff_deg_i pour calculer le coefficient de la polynôme PQ de degré i         
        for j in range(i+1):                          
            if j<=d_p : coeff_P_deg_j=P[j]
            else:coeff_P_deg_j=0             
            if (i-j)<=d_q : coeff_Q_deg_j=Q[i-j]
            else: coeff_Q_deg_j=0             
            coeff_PQ_deg_i += coeff_Q_deg_j*coeff_P_deg_j
        PQ[i]=coeff_PQ_deg_i
    return PQ[:7]

def code(P):
    return modulo(produit_polynome(g,P))


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

   
L = [ (Mots77[k], code(Mots77[k])) for k in range(len(Mots77)) ]

A = evalue([0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0])
    
def decode (P):
    A = simplifie(evalue(P))
    if A[0] == -1 :
        return P
    else : 
        P[A[1]] = (1 + P[A[1]])%2 
    return P

def int_to_bin(n):
    l = []
    q = - 1 
    r = ''
    while n > 0 :
        r = n%2
        l.append(r)
        n = (n-r)/2
    k = len(l) 
    A = l + [0 for k in range(7 - len(l))]
    return A 

def text_to_bin(c):
    n = len(c)
    l = []
    for k in range(n):
        l.append(code(int_to_bin(int(c[k]))))
    return l
    
def aleatoire(t):
    n = len(t)
    a = rd.randint(0,n-1)
    t[a] = (t[a] + 1)%2 
    return t
    
    
def switch(c):
    a = text_to_bin(c)
    n = len(a)
    for k in range(n):
        a[k] = aleatoire(a[k])
    return a


#Correction souvent bonne mais parfois fausse d'un même mot de code 

def egal (A,B):
    n = len(A)
    for k in range(n):
        if not (A[k] == B[k]) :
            return False
    return True 

def associe(A):
    k = 0 
    while not (egal(L[k][1],A)):      #Associe le mot d'origine au mot de code trouvé 
        k += 1 
    return L[k][0]

def flaten(l):
    n = len(l)
    c = ''
    for k in range(n):
        l[k] = str(l[k])
        c += l[k]
    return c


def final(t):
    l = []
    for k in range(len(t)):
        l.append(bin_to_int(associe(decode(t[k]))))
    return flaten(l)



def bin_to_int(t):
    n = len(t)
    acc = 0
    for k in range(n):
        acc += t[k]*(2**k) #n-k-1
    return acc

def bin_to_text (t):
    n = len(t)
    c = []
    for k in range(n):
        c.append(int(bin_to_int(t[k])))
    return flaten(c)

def envoie(c):
    s = switch(c)
    return print('le numéro reçu est',bin_to_text(s),"mais a été décodé et j'ai trouvé :", final(s))





import PIL.Image as im
import matplotlib.pyplot as plt

def envoie2(c):
    s = switch(c)
    return bin_to_text(s),final(s)

def Pixels(fichier):
    image = im.open(fichier)
    pixels = np.array(image)
    return pixels
    
def Afficher(pixels):
    plt.imshow(pixels, cmap = 'Greys_r')
    plt.show()
    
Gandalf = Pixels('gandalf.jpg')
    
def flaten2(t):
    n = len(t)
    l = np.zeros(n)
    for k in range(n):
        l[k] = t[k,0]
    return l
    
    
    
def varie(T):
    n,m,p = T.shape
    U = np.copy(T)
    N = np.copy(U)
    for k in range(n):
        for j in range (m):
            for i in range(p):
                fake, truth = envoie2(str(T[k,j,i]))
                U[k,j,i] = int(fake)%255
                N[k,j,i] = int(truth)
    return U, N 

```
