import numpy as np 
import sys 
import random as rd
from mat import *
import time


def In_Hamm(m):
    Y = module1(np.dot(H7,m))
    n = len(Y)
    for i in range(n):
        if not (Y[i]) == 0 :
            return False 
    return True 

def In_class(m,Z):
    n = len(Z)
    U = module1(np.dot(H7,m))
    for i in range(n-1):
        if not (U[i] == Z[i]) :
            return False 
    return True 

def egal (A,B):
    n = len(A)
    for k in range(n):
        if not (A[k] == B[k]) :
            return False
    return True 

def somme(m):
    n,acc = len(m), 0
    for k in range(n):
        acc += m[k]
    return acc

 
############################

def pot_somme(C,p):                       #Toutes les sommes de p colonnes de C
    m, n = np.shape(C)
    def sommes(p):
        if p==0: 
            return [[np.zeros((1,3))[0],[-1]]]      #Initialise à -1 pour colonnes
        else : 
            s = sommes(p-1)
            for j in range(len(s)):
                for k in range(m):
                    s.append([module1(s[j][0]+ C[k]),s[j][1] + [k]])
            return s
    return sommes(p-1)

############################    

def laterale(U):                        #Détermine classe latérale associée à U
    Z = module1(np.dot(H7,U))
    p, q = np.shape(H7)
    lat = []
    for k in range(len(All_words)):
        l = All_words[k]
        if In_class(l,Z) :                  
            lat.append((somme(l),l))
    return lat

def min_laterale(U):                   # Détermine le min de la classe latérale
    lat = laterale(U)
    n = len(lat)
    (min, associe) = (lat[0][0],lat[0])          
    for k in range(1,n):
        l, u = lat[k]
        if l < min and l > 0 :   
            associe = lat[k][1]
    return associe

###########################

def decodage_1(m):                      #Décode première méthode non optimisée  
    if In_Hamm(m):
        return module1(m)
    else:    
        S = module1(np.dot(H7,m))
        M = min_laterale(m)
        p = somme(M)
        acc = 0 
        Somme_colonnes = pot_somme(H77,p)
        while not(egal(Somme_colonnes[acc][0],S)):        
                acc += 1        
    A = module1(m)
    for i in Somme_colonnes[acc][1] :
        if i >= 0 :
            A[i] = (A[i] + 1)%2          # + E_i
    k = 0
    while not (egal(L[k][1],A)):      #Associe le mot d'origine au mot de code trouvé 
        k += 1 
    return L[k][0]

#####################################


def int_to_bin(n):
    l = []
    q = - 1 
    r = ''
    while n > 0 :
        r = n%2
        l.append([r])
        n = (n-r)/2
    k = len(l) 
    l.reverse()
    return np.array([[0] for i in range(4-k)] + l)

def bin_to_int(t):
    n = len(t)
    acc = 0
    for k in range(n):
        acc += t[n-k-1]*(2**k)
    return int(acc)

def aleatoire(t):
    n = len(t)
    a = rd.randint(0,n-1)
    t[a] = (t[a] + 1)%2 
    return t

def text_to_bin(c):
    n = len(c)
    l = []
    for k in range(n):
        l.append(module1(np.dot(G7,int_to_bin(int(c[k])))))
    return l
    
def switch(c):
    a = text_to_bin(c)
    n = len(a)
    for k in range(n):
        a[k] = aleatoire(a[k])
    return a

def flaten(l):
    n = len(l)
    c = ''
    for k in range(n):
        l[k] = str(l[k])
        c += l[k]
    return c
    
    
def bin_to_text (t):
    n = len(t)
    c = []
    for k in range(n):
        c.append(int(bin_to_int(t[k])))
    return flaten(c)

def final (t):
    c = []
    n = len(t)
    for k in range(n):
        c.append(bin_to_int(decodage_1(t[k])))
    return flaten(c)

def envoie(c):
    s = switch(c)
    return print('Le numéro reçu est',bin_to_text(s),"mais a été décodé par :", final(s))

############################################@


def envoie2(c):
    s = switch(c)
    return bin_to_text(s),final(s)
    

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
    


####################################

PotH77 = pot_somme(H77,2)

def creer_erreur(L):
    n = len(L)
    e = [0]*7
    for k in range(n):
        if L[k] >= 0 :
            e[L[k]] = 1 
    return e

Mots_erreur = [creer_erreur(PotH77[k][1]) for k in range(len(PotH77))]

Liste_somme = [(PotH77[k][0],Mots_erreur[k]) for k in range(len(PotH77))]
Dico_somme = dict([(str(PotH77[k][0]),Mots_erreur[k]) for k in range(len(PotH77))]) 


####################################

def decodage_2(m):                      #Décode 
    if In_Hamm(m):
        return module1(m)
    else:    
        S = module1(np.dot(H7,m))
        acc = 0 
        e = Dico_somme.get(str(S))      
    return module1(m + e)

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

def switch2(k):
    a = int_to_bin2(k)
    U,V = module1(np.dot(G7,a[:4])), module1(np.dot(G7,a[4:]))
    return aleatoire(U), aleatoire(V)

def code(k):
    a = int_to_bin2(k)
    U,V = module1(np.dot(G7,a[:4])), module1(np.dot(G7,a[4:]))
    return U,V

def final2 (U,V):
    A, B = decodage_2(U), decodage_2(V)
    C, D = Dictio.get(str(bin_to_int(A))), Dictio.get(str(bin_to_int(B)))
    Z = np.concatenate((C,D))
    return bin_to_int(Z)

def envoie3(L):  
    if rd.random() < 0.5:
        s,v = switch2(L)
        a = s + v
        return bin_to_int(a),final2(s,v)
    else : 
        u,v = code(L)
        b = final2(u,v)
        return b, b

def varie2(T):
    n,m,p = T.shape
    U = np.copy(T)
    N = np.copy(U)
    for k in range(n):
        for j in range (m):
            for i in range(p):
                fake, truth = envoie3(T[k,j,i])    
                U[k,j,i] = fake%255
                N[k,j,i] = truth
    return U, N 



#t0 = time.time()
    
#Fake, Truth = varie2(Gandalf)

#print("Temps d execution : %s secondes ---" % (time.time() - t0))



