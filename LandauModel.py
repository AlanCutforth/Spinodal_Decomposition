import numpy as np
import matplotlib.pyplot as plt
import random as random
from scipy.fft import fft

# Calculates the local hamiltonian for a lattice point.
def hamiltonian(s,i,j,l):
    H = s[(i+1)%l,j] + s[i,(j+1)%l] + s[(i-1)%l,j] + s[i,(j-1)%l]
    
    return H

# The following 2 functions slice the lattice into horizontal and vertical strips,
# and calculate the average length of sequences of negative/positive states across
# all of these strips.

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
        if ls[1,k] != 0:
            lengths.append(ls[0,k])
                        
    return len(lengths), sum(lengths)
    

def avg_length(s, size):
    l = 0
    q = 0
    mod_s = sign_matrix(s)
    
    for i in range(size):
        indices, summed = avg_string(mod_s[i,:])
        l += summed
        q += indices
        
        indices, summed = avg_string(mod_s[:,i])
        l += summed
        q += indices
        
        
    return l/q

# Returns a new lattice with only whether each point on the Landau lattice is
# positive or negative.
def sign_matrix(s):
    sign = np.zeros((size,size))
    
    for i in range(size):
        for j in range(size):
            if s[i,j] != 0:
                sign[i,j] = s[i,j]/np.abs(s[i,j])
                
    return sign

random.seed(5)

size = 100
steps = 1000

dt = 0.1

model_plot = plt.figure()
model_plot.add_axes()
plt.title("Conserved Field Landau Grid")

struc_fac = []
time = []
L = []

# Sets up random initial lattice
s = np.zeros((size,size))
for i in range(size):
    for j in range(size):
        if random.randint(0,1) == 0:
            s[i,j] = 1
        else:
            s[i,j] = -1

# Performs the Monte-Carlo simulation on the Landau equation.
for t in range(steps):
    sf = np.zeros((size,size))
    
    for i in range(size):
        for j in range(size):
            ds = 0
            ds = dt*(hamiltonian(s,i,j,size) - 5*s[i,j] + s[i,j]**3)
            sf[i,j] = s[i,j] + ds
            
    for i in range(size):
        for j in range(size):
            s[i,j] = sf[i,j]
            
    fourier = fft(s)
    func = np.zeros((size,size))
    for i0 in range(size):
        for j0 in range(size):
            func[i0,j0] = abs(fourier[i0,j0]*np.conj(fourier[i0,j0]))
            
    struc_fac.append(np.average(func))
    time.append(t*dt)
    L.append(avg_length(s,size))
            
    if t % 100 == 0:          
        model_plot = plt.imshow(s,interpolation='nearest')
        plt.pause(0.5)
    
    print(t)

# Plots the graphs
fig3 = plt.figure()
plt.title("Structure Factor")
plt.ylabel("Structure Factor")
plt.xlabel("Time Steps")
plt.plot(time, struc_fac)

fig4 = plt.figure()
plt.title("Log Plot of Structure Factor")
plt.ylabel("Log of Structure Factor")
plt.xlabel("Log of Time Steps")
plt.plot(np.log(time)/time, np.log(struc_fac)/time)

fig5 = plt.figure()
plt.title("Length Scale vs. Steps (Time)")
plt.xlabel("Steps/100 (Time)")
plt.ylabel("Charactersitic Length Scale")
plt.plot(time, L, '.')

fig6 = plt.figure()
plt.title("Logarithmic Length Scale vs. Steps (Time)")
plt.xlabel("Log of Steps/100 (Time)")
plt.ylabel("Log of Charactersitic Length Scale")
plt.plot(np.log(time), np.log(L), '.')

            

    
    