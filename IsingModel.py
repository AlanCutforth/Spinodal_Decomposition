import numpy as np
import matplotlib.pyplot as plt
import random as random
from numpy.random import rand
from scipy.fft import fft

# Calculates the local hamiltonian for a lattice point.
def hamiltonian(s,i,j,J,l):
    H = s[(i+1)%l,j] + s[i,(j+1)%l] + s[(i-1)%l,j] + s[i,(j-1)%l]
    
    return -1*J*H*s[i,j]

# The following 2 functions slice the lattice into horizontal and vertical strips,
# and calculate the average length of sequences of same-spin states across all of
# these strips.

def avg_string(A):
    ls = np.zeros((2,len(A)))
    n = 0
    lengths = []
    
    for i in range(len(A)):
        if i == 0:
            ls[1,0] = A[i]
            ls[0,0] = 1
        else:
            if A[i] == A[i-1]:
                ls[0,n] += 1
            else:
                n += 1
                ls[1,n] = A[i]
                ls[0,n] += 1
                
        if i == len(A) - 1:
            if A[0] == A[i]:
                ls[0,0] += ls[0,n]
                ls[0,n] = 0
                ls[1,n] = 0
                
    for k in range(len(A)):
        if ls[0,k] != 0:
            lengths.append(ls[0,k])
                        
    return len(lengths), sum(lengths)

    

def avg_length(s, size):
    l = 0
    q = 0
    
    for i in range(size):
        indices, summed = avg_string(s[i,:])
        l += summed
        q += indices
        
        indices, summed = avg_string(s[:,i])
        l += summed
        q += indices
        
        
    return l/q
            
# Calculates the sum of all the Ising lattice points
def lattice_sum(s, length, b_in=0):
    s_sum = 0
    b = int(b_in)
    
    for i in range(b,length-b):
        for j in range(b,length-b):
            s_sum += s[i,j]
            
    return s_sum

# Returns the k-space vector of two points.
def k_space(i, j):
    i=i+1
    j=j+1
    return np.sqrt(1/i**2 + 1/j**2)


## Parameters
isBound = True
size = 128
steps = 200000
energy = 1
temp = 0

k_struc = []
time = []

w = 2

random.seed(5)

Tc = 2.26918531421*energy


X = list(range(0,size))
Y = list(range(0,size))

# Sets up the initial random Ising lattice.
s = np.zeros((size,size))
for i in range(size):
    for j in range(size):
        if random.randint(0,1) == 0:
            s[i,j] = 1
        else:
            s[i,j] = -1
            

model_plot = plt.figure()
model_plot.add_axes()
plt.title("2D Ising Model (Final)")
axf = plt.gca()
model_plot = plt.imshow(s,interpolation='nearest')

n_ups = []
m = []

L = []
number = 0

# Performs the Ising model Monte-Carlo simulation.
for i in range(steps):
    flip_i = np.random.randint(0,size)
    flip_j = np.random.randint(0,size)
    
    preflip_sij = s[flip_i, flip_j]
    old_H = hamiltonian(s, flip_i, flip_j, energy, size)
    s[flip_i, flip_j] = s[flip_i, flip_j]*-1

    dE = hamiltonian(s, flip_i, flip_j, energy, size) - old_H
    
    if dE > 0:
        s[flip_i, flip_j] = preflip_sij
        
        if rand() < np.exp(-dE/temp):
            s[flip_i, flip_j] = s[flip_i, flip_j]*-1
        
        print(i)
    
    if i % 500 == 0:
        L.append(avg_length(s,size))
        
        fourier = fft(s)
        struc_fac = fourier*np.conj(fourier)
        
        S_struc_fac = 1
        KS_struc_fac = 0

        k_matrix = np.zeros((int(size/2),int(size/2)))
        for i_index in range(int(size/2)):
            for j_index in range(int(size/2)):
                k_matrix[i_index, j_index] = k_space(i_index, j_index)
            

        if S_struc_fac != 0:
            k_struc.append(np.average(struc_fac[0:int(size/2),0:int(size/2)], weights=k_matrix))#KS_struc_fac/S_struc_fac)
        else:
            k_struc.append(0)
        
        number += 1
        time.append(number)
        
        model_plot = plt.imshow(s,interpolation='nearest')
        plt.pause(0.5)
        

# Plots the graphs.
model_plot = plt.imshow(s,interpolation='nearest')

fig2 = plt.figure()
plt.title("Length Scale vs. Steps (Time)")
plt.xlabel("Steps/100 (Time)")
plt.ylabel("Charactersitic Length Scale")
plt.plot(time, L, '.')

fig3 = plt.figure()
plt.title("Logarithmic Length Scale vs. Steps (Time)")
plt.xlabel("Log of Steps/100 (Time)")
plt.ylabel("Log of Charactersitic Length Scale")
plt.plot(np.log(time), np.log(L), '.')

fig4 = plt.figure()
plt.title("Average K-Value vs. Steps (Time)")
plt.xlabel("Steps/100 (Time)")
plt.ylabel("K-Value")
plt.plot(time, k_struc, '.')

fig5 = plt.figure()
plt.title("Logarithmic K-Value vs. Steps (Time)")
plt.xlabel("Log of Steps/100 (Time)")
plt.ylabel("Log of K-Value")
plt.plot(np.log(time), np.log(k_struc), '.')

fig6 = plt.figure()
plt.title("Structure Factor vs k")
plt.xlabel("Log of Steps/100 (Time)")
plt.ylabel("Log of K-Value")
plt.plot(np.log(time), np.log(k_struc), '.')

