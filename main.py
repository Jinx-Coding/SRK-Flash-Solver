# SRK Flash Equilibrium solver - Giovanni Correra 10/2023 #

# Libraries import #

import numpy
import numpy as np
from scipy.optimize import fsolve

# Data #

precision = 1e-20  # Setting the Ratchford Rice zero function precision #

zF = np.array((0.013148, 0.020329, 0.00079, 0.965733)) # Feed molar fractions array (-) #
F = 2559.96 # Feed molar flowrate (kmol/d) #
F = zF * F
P = 1.01325 # System pressure (bar) #
T = 308.15 # System temperature (K) #

Tc = np.array([190.4, 304.1, 373.2, 647.3]) # Critical temperatures array (K) #
Pc = np.array([46, 73.8, 89.4, 221.2]) # Critical pressure array (bar) #
om = np.array([0.011, 0.239, 0.081, 0.344]) # Omega array (-) #

# Data check #

if sum(zF) != 1 :
    print('ERROR : feed molar fractions do not add up to 1')
    exit()

# First guess values #

index = np.where(Tc == max(Tc))
alpha_fg = (sum(F)-F[index])/sum(F) # First guess alpha (defined as V/F), assuming only heaviest component to be liquid
V_fg = alpha_fg * sum(F)

x_fg = np.zeros(len(Tc))
x_fg[index] = 1

y_fg = F/V_fg
y_fg[index] = 0

control = 0 # If needed control = 1 assures solution stability as alpha = 0.5 and y_fg, x_fg = n (1/n) #

if control == 1 :
    alpha_fg = 0.5
    y_fg = np.full(len(Tc),1/len(Tc))
    x_fg = np.full(len(Tc),1/len(Tc))

# Functions #

# SRK Solving function #
def SRK(T,P,Tc,Pc,om,y,phase) :

    # SRK solving function with basic mixture laws #
    # a(i, j) = (a(i) * a(j)) ^ 0.5 #
    # b(i, j) = (b(i) + b(j)) / 2 #

    # Phase = 1(vapour), Phase = 2(liquid) #

    R = 8.3145
    RT = R*T
    RTc = R*Tc

    S = 0.48 + 1.574 * om -0.176*om**2
    k = (1 + S * (1-np.sqrt(T/Tc)))**2
    a = (0.42748*k*RTc**2)/Pc
    b = 0.08664*RTc/Pc
    AS = a*P/RT**2
    BS = b*P/RT
    aM = np.sqrt(np.array([a]).T * a)
    bM = np.zeros((len(Tc),len(Tc)))

    for i in range(len(Tc)) :
        for j in range(len(Tc)) :
            bM[i,j] = (b[i]+b[j])/2

    am = y * aM * np.array([y]).T
    am = np.sum(am)

    bm = y * bM * np.array([y]).T
    bm = np.sum(bm)

    A = am*P/RT**2
    B = bm*P/RT

    alfa = -1
    beta = A-B-B**2
    gamma = -A*B

    # Analytic solution #

    p = beta - (alfa**2)/3
    q = 2*(alfa**3)/27 - alfa*beta/3 + gamma
    q2 = q/2
    a3 = alfa/3
    D = (q**2)/4 + (p**3)/27

    if D > 0 :
        Z1 = np.power(-q2+np.sqrt(D),1/3) + np.power(-q2-np.sqrt(D),1/3) - a3
        Z = np.array((Z1, Z1, Z1))
    elif D == 0 :
        Z1 = -2*np.power(q2,1/3) - a3
        Z2 = np.power(q2,1/3) - a3
        Z = np.array((Z1, Z2, Z2))
    elif D < 0 :
        r = np.sqrt((-p**3)/27)
        teta = np.arccos(-q2*np.sqrt(-27/p**3))
        Z1 = 2*np.power(r,1/3)*np.cos(teta/3) - a3
        Z2 = 2*np.power(r,1/3)*np.cos((2*np.pi+teta)/3) - a3
        Z3 = 2*np.power(r,1/3)*np.cos((4*np.pi+teta)/3) - a3
        Z = np.array((Z1, Z2, Z3))

    if phase == 1 :
        Z = max(Z)
    elif phase == 2 :
        Z = min(Z)

    return(Z, AS, BS, A, B)


# SRK fugacity coefficient calculation function #

def fugacity(Z, AS, BS, A, B) :

    # Takes SRK parameters and determines mixture vapour and liquid fugacities #

    lnphi = (Z-1)*BS/B + (A/B)*((BS/B)-2*np.sqrt(AS/A))*np.log((Z+B*(1+np.sqrt(2)))/(Z+B*(1-np.sqrt(2))))-np.log(Z-B)
    phi = np.exp(lnphi)

    return(phi)


# Ratchford - Rice zero function #

def RR(alpha, zF, K) :

    single = zF * (K-1) / (1 + alpha * (K-1))
    fun = np.sum(single)

    return(fun)


# Main script #

error = 1
j = 0

while error > precision :
    j = j + 1
    (ZV, AS, BS, AV, BV) = SRK(T, P ,Tc, Pc, om, y_fg, 1)
    (ZL, _, _, AL, BL) = SRK(T, P, Tc, Pc, om, x_fg,2)

    phiV = fugacity(ZV, AS, BS, AV, BV)
    phiL = fugacity(ZL, AS, BS, AL, BL)

    K = phiL/phiV

    alpha = fsolve(lambda x : RR(x, zF, K), alpha_fg)
    x = zF/(1+alpha*(K-1))
    y = K*zF/(1+alpha*(K-1))

    error = np.abs(alpha-alpha_fg)

    alpha_fg = alpha
    x_fg = x
    y_fg = y


# Post - processing

print('N iter =', j)
print('alpha = ', end="")
print('%.12f' % np.squeeze(alpha))

for i in range(len(Tc)) :
   print('y = ', end="")
   print('%.12f' % y[i], end=" ")

print('')

for i in range(len(Tc)) :
    print('x = ', end="")
    print('%.12f' % x[i], end=" ")
