# LIBRERIAS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from time import time
from joblib import Parallel, delayed


seed_value = 115

# Seed Numpy
np.random.seed(seed_value)
rng = np.random.default_rng(seed_value)



# MHS


'''
Las siguientes funciones estaran en formato f(A,hparams) donde A en el alineamiento actual (ver MHS) y 
hparams los hiperparametros fijos del problema, donde hparams tiene la forma (Seq,w,fondo) donde Seq es el arreglo de N
strings de tamanio M donde se busca el motif, w el ancho del motif y fondo las probabilidades de background
'''


#  Alineamiento aleatorio


#alineador: {tamano de motif} x {num. de seq.} x {Ancho de seq.}

def alineador(hparams):
    Seq = hparams[0]
    w=hparams[1]
    fondo = hparams[2]
    N=len(Seq)
    M=len(Seq[0])
    return rng.uniform(0,M-w,size=N).astype(int)


#  PWM


#Matriz de Position Weight Matrix
def PWM(A,hparams):
    #vars
    Seq = hparams[0]
    w=hparams[1]
    N=len(Seq)

    # Convertimos la informacion a un 2d-nparray
    data = np.array([list(Seq[i][A[i]:A[i]+w]) for i in range(N)])

    # Definimos el vocabulario y la matriz
    alpha = ["A","C","G","T"]
    Matrix=np.zeros((w,4))
    
    # Calculamos frecuencias con los respectivos margenes
    for i in range(4):
        mask = (data == alpha[i])
        Matrix[:, i] += mask.sum(axis=0)+(1.25/4)

    # Normalizamos
    Matrix = np.divide(Matrix, N+ 1.25)
    return Matrix


# Energía
def H(A,hparams,pesos=None):
    #vars
    Seq = hparams[0]
    w=hparams[1]
    fondo = hparams[2]
    N=len(Seq)
    M=len(Seq[0])

    if pesos is None:
        Pesos=PWM(A,hparams)
    else:
        Pesos=pesos
    fondos=np.tile(fondo,(w,1))

    return np.sum(Pesos*np.log(Pesos))*N


# Densidades de probabilidad

def scan(pos,index,pesos,hparams):
    #Vars
    Seq = hparams[0]
    w=hparams[1]
    fondo = hparams[2]
    N=len(Seq)
    M=len(Seq[0])
    alpha = ["A","C","G","T"]
    
    suma=0
    
    for j in range(pos,pos+w):
        for k in range(4):
            suma+=np.log(pesos[j-pos,k]/fondo[k])*(Seq[index][j]==alpha[k])
                
    return max(suma,0.001) 


def pitatoria(positions,A,hparams):
    #Vars
    Seq = hparams[0]
    w=hparams[1]
    fondo = hparams[2]
    N=len(Seq)
    M=len(Seq[0])
    Pesos=PWM(A,hparams)

    pit=1
    for i in range(N):
        pit=pit*(scan(positions[i],i,Pesos,hparams))
    return pit



#  Sampleo

def samp(A,hparams):
    #Vars
    Seq = hparams[0]
    w=hparams[1]
    fondo = hparams[2]
    N=len(Seq)
    M=len(Seq[0])
    Pesos=PWM(A,hparams)
    
    # Calcula pesos
    B=np.zeros(N,dtype=int)
    P=np.zeros((N,M-w))
    norm=np.zeros(N)
    
    for i in range(N):
        # Escanea cada punto 
        for j in range(M-w):
            P[i,j]=scan(j,i,Pesos,hparams)
        # Obtiene constantes de normalizacion
        norm[i]=1/np.sum(P[i,:])
    
    # Normaliza    
    P=np.diag(norm) @ P
    
    # Samplea
    for i in range(N):
        B[i]= rng.choice(M-w,p=P[i,:])

    return B


# Implementación MHS

def MHS(a0,n,lam,hparams,OutputFlag=False):
    #Vars
    Seq = hparams[0]
    w=hparams[1]
    fondo = hparams[2]
    N=len(Seq)
    M=len(Seq[0])
    
    A= np.zeros(n,dtype=object)
    A[0]=a0
    
    for i in range(n-1):
        t0=time()
        # Proponemos un candidato
        B = samp(A[i],hparams)
        
        t1=time()
        
        # Aceptamos-rechazamos el candidato
        samples1= pitatoria(B,A[i],hparams)
        samples2= pitatoria(A[i],B,hparams)
        
        paso =min(1,np.exp(lam*(H(B,hparams)-H(A[i],hparams))*samples1/samples2))

        if rng.uniform(0,1) < paso:
            A[i+1] = B.copy()
            if OutputFlag:
                print("candidato aceptado","iter:",i+1,"sample time:",t1-t0,"A-R time:",time()-t1)
        else:
            A[i+1] = A[i].copy()
            if OutputFlag:
                print("candidato rechazado","iter:",i+1,"sample time:",t1-t0,"A-R time:",time()-t1)
    return A



# IMC

def IMC(n,R,lam,hparams):
    #Vars
    #Seq = hparams[0]
    #w=hparams[1]
    #fondo = hparams[2]
    #N=len(Seq)
    #M=len(Seq[0])

    parametros=[]
    for i in range(R):
        a0=alineador(hparams)
        parametros.append((a0,n,lam,hparams))
    
    #paralelizar
    cadenas = Parallel(n_jobs=-1)(delayed(MHS)(*args) for args in parametros)
    
    alineamientos = list(cadenas)
    alineamientos.sort(reverse=True,key=lambda A:H(A[-1],hparams))

    return alineamientos


#  PMC

# Funciones aux necesarias

# Densidades de probabilidad PMC
def PMC_scan(pos,index,R_pesos,hparams):
    # Guardamos los R scans
    R=len(R_pesos)
    probas=np.zeros(R)
    
    # Calculamos
    for i in range(R):
        probas[i] = scan(pos,index,R_pesos[i],hparams)

    return probas.mean()

def PMC_pitatoria(positions,cadenas,hparams):
    #Vars
    Seq = hparams[0]
    N=len(Seq)
    R=len(cadenas)
    R_pesos=[PWM(cadenas[k],hparams) for k in range(R)]

    pit=1
    for i in range(N):
        pit *= PMC_scan(positions[i],i,R_pesos,hparams)
    
    return pit 

# Funcion que samplea
def PMC_samp(cadenas,hparams):
    # Vars
    Seq = hparams[0]
    w=hparams[1]
    fondo = hparams[2]
    N=len(Seq)
    M=len(Seq[0])
    R=len(cadenas)
    R_pesos=[PWM(cadenas[k],hparams) for k in range(R)]
    
    """
    # Codigo sin paralelizar:
    # Lista de candidatos a samplear
    candidatos =[]
    
    # Loop que recorre las cadenas
    for _ in range(R):
        # Calcula pesos
        B=np.zeros(N,dtype=int)
        P=np.zeros((N,M-w))
        norm=np.zeros(N)
        
        for i in range(N):
            # Escanea cada punto 
            for j in range(M-w):
                P[i,j]=PMC_scan(j,i,R_pesos,hparams)

            # Obtiene constantes de normalizacion
            norm[i]=1/np.sum(P[i,:])
        
        # Normaliza    
        P=np.diag(norm) @ P
        
        # Samplea
        for i in range(N):
            B[i]= rng.choice(M-w,p=P[i,:])

        # Agrega a los candidatos
        candidatos.append(B)
    """
    def subsamp():
        # Calcula pesos
        B=np.zeros(N,dtype=int)
        P=np.zeros((N,M-w))
        norm=np.zeros(N)
        
        for i in range(N):
            # Escanea cada punto 
            for j in range(M-w):
                P[i,j]=PMC_scan(j,i,R_pesos,hparams)

            # Obtiene constantes de normalizacion
            norm[i]=1/np.sum(P[i,:])
        
        # Normaliza    
        P=np.diag(norm) @ P
        
        # Samplea
        for i in range(N):
            B[i]= rng.choice(M-w,p=P[i,:])

        return B
    candidatos = Parallel(n_jobs=-1)(delayed(subsamp)() for k in range(R))
    return candidatos


# Implementación PMC

def PMC(n,R,lam,hparams):
    #Vars
    Seq = hparams[0]
    w=hparams[1]
    fondo = hparams[2]
    N=len(Seq)
    M=len(Seq[0])

    # Funcion para extraer el i-esimo paso de las R-cadenas
    def paso(cadenas,index):
        frame= []
        for i in range(len(cadenas)):
            frame.append(cadenas[i][index].copy())
        return frame

    # Lista donde se guardaran las R cadenas simuladas
    alineamientos=[]
    for i in range(R):
        alineamientos.append(np.zeros(n,dtype=object))
        alineamientos[i][0]= alineador(hparams)
    
    # For loop del algoritmo
    for i in range(n-1):
        # Sampleamos candidatos de las distintas cadenas una lista de R candidatos 

        candidatos = PMC_samp(paso(alineamientos,i),hparams)

        # Aceptamos-rechazamos
        for j in range(R):
            # Cuociente de probas de paso
            Q1= PMC_pitatoria(candidatos[j],paso(alineamientos,i),hparams)
            Q2= PMC_pitatoria(paso(alineamientos,i)[j],candidatos,hparams)
            Q = Q1/Q2

            # Diff. de energia
            E_diff= H(candidatos[j],hparams) - H(alineamientos[j][i],hparams)

            # Calculo de paso de aceptacion
            alfa=min(1,np.exp(E_diff)*Q)

            if rng.uniform() < alfa:
                alineamientos[j][i+1]=candidatos[j].copy()
            else:
                alineamientos[j][i+1]=alineamientos[j][i].copy()

    #ordenamiento final de la mejor cadena 
    alineamientos.sort(reverse=True,key=lambda A:H(A[-1],hparams))
    
    return alineamientos


# Batches

def BATCHER(n,R,lam,prop,hparams):
    # prop=Proporción del batch
    new=hparams[0].copy()

    nuevo=int(len(new)*prop)
    nuevoparams=[new[:nuevo],hparams[1],hparams[2]]
    A = IMC(n,R,lam,nuevoparams)[0][-1]
    
    motivo=new[0][A[0]:A[0]+hparams[1]]
    restantes=np.zeros(len(new)-nuevo)
    i=0
    j=1
    while i < len(restantes):
        found=new[nuevo+i].find(motivo)
        if found != -1:
            restantes[i]=found
            i+=1
        else:
            print(j)
            j+=1
            A = IMC(n,R,lam,nuevoparams)[0][-1]
            motivo=new[0][A[0]:A[0]+hparams[1]]
            i=0
        
    out=np.concatenate((A,restantes)).astype(int)

    return out

def BATCHER2(n,R,lam,prop,hparams):
    # prop=Proporción del batch
    new=hparams[0].copy()

    nuevo=int(len(new)*prop)
    nuevoparams=[new[:nuevo],hparams[1],hparams[2]]
    A = IMC(n,R,lam,nuevoparams)[0][-1]
    
    def pwm_samp(Pesos):
        w=Pesos.shape[0]
        motif=""
        for i in range(w):
            motif += rng.choice(["A","C","G","T"],p=Pesos[i,:])
        return motif
        
    motivo=pwm_samp(PWM(A,nuevoparams))
    restantes=np.zeros(len(new)-nuevo)
    
    i=0
    j=1
    while (i < len(restantes)) and (j<5*len(new)):
        found=new[nuevo+i].find(motivo)
        if found != -1:
            restantes[i]=found
            i+=1
        else:
            print(j)
            j+=1
            motivo=pwm_samp(PWM(A,nuevoparams))
        
    out=np.concatenate((A,restantes)).astype(int)

    return out


def BATCHER_PMC(n,R,lam,prop,hparams):
    # prop=Proporción del batch
    new=hparams[0].copy()

    nuevo=int(len(new)*prop)
    nuevoparams=[new[:nuevo],hparams[1],hparams[2]]
    A = PMC(n,R,lam,nuevoparams)[0][-1]
    
    motivo=new[0][A[0]:A[0]+hparams[1]]
    restantes=np.zeros(len(new)-nuevo)
    i=0
    j=1
    while i < len(restantes):
        found=new[nuevo+i].find(motivo)
        if found != -1:
            restantes[i]=found
            i+=1
        else:
            print(j)
            j+=1
            A = PMC(n,R,lam,nuevoparams)[0][-1]
            motivo=new[0][A[0]:A[0]+hparams[1]]
            i=0
        
    out=np.concatenate((A,restantes)).astype(int)
    return out

# RWMFA(Random Walk Motif Finding Algorithm)

def RW(A,hparams):
    #Vars
    Seq = hparams[0]
    w=hparams[1]
    fondo = hparams[2]
    N=len(Seq)
    M=len(Seq[0])
    
    B= A.copy() + rng.choice(np.arange(-w,w),size=N)
    
    def fixer(l):
        if l>M-w:
            return rng.choice(np.arange(M-2*w,M-w))
        elif l<0:
            return rng.choice(np.arange(0,w))
        else:
            return l
    
    for i in range(len(B)):
        B[i]=fixer(B[i])
    return B
    
def RWMFA(a0,n,lam,hparams,OutputFlag=False):
    #Vars
    Seq = hparams[0]
    w=hparams[1]
    fondo = hparams[2]
    N=len(Seq)
    M=len(Seq[0])
    
    A= np.zeros(n,dtype=object)
    A[0]=a0
    
    for i in range(n-1):
        t0=time()
        # Proponemos un candidato
        B = RW(A[i],hparams)
        
        t1=time()

        H_diff= H(B,hparams)-H(A[i],hparams)   
        paso =min(1,np.exp(lam*H_diff))

        if rng.uniform(0,1) < paso:
            A[i+1] = B.copy()
            if OutputFlag:
                print("candidato aceptado","iter:",i+1,"sample time:",t1-t0,"A-R time:",time()-t1)
        else:
            A[i+1] = A[i].copy()
            if OutputFlag:
                print("candidato rechazado","iter:",i+1,"sample time:",t1-t0,"A-R time:",time()-t1)
    return A


# # Escritura de experimento

"""
def write_arrays_to_file(arrays, filename):
    with open(filename, 'w') as f:
        for i, array in enumerate(arrays):
            # Convert array to space-separated string
            array_str = ' '.join(map(str, array))
            f.write(array_str + '\n')
            # Add * symbol between lines (except after the last array)
            if i < len(arrays) - 1:
                f.write('*\n')
"""
