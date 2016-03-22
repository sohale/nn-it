#Also see http://live.sympy.org/

from __future__ import division
from sympy import *
x, y, z, t = symbols('x y z t')
k, m, n = symbols('k m n', integer=True)
f, g, h = symbols('f g h', cls=Function)
f = tanh(x)
#f
#tanh(x)
#tanh⁡(x)

print( diff(f) )
#−tanh2(x)+1
#−tanh2⁡(x)+1

#g = x1+x2

x1,x2 = symbols('x1 x2')
print( x1 )

x1,x2,x3,x4,x5 = symbols('x1 x2 x3 x4 x5')

w11,w12,w21,w22 = symbols('w11 w12 w21 w22')
y1 = tanh(x1 * w11 + x2 * w21)
print(y1)
#tanh(w11x1+w21x2)

print( diff(y1, x1) )
#w11(−tanh2(w11x1+w21x2)+1)

print( diff(y1, x2) )
#w21(−tanh2(w11x1+w21x2)+1)

print( diff(y1, x3) )
# 0

#layers = [3,4,2]
layers = [3, 2]


#contains sympy variables
#for i in range(M):
#    for j in range(N):
#        print(vars_x[i][j], end=", ")
#    print()

# prepare the space of variables
#vars_x = [[None, None], [None,None], [None, None]] 
vars_x = []
#M = layers[0]
#N = layers[1]
L = len(layers)
for l in range(L):
    vars_x.append([])
    for i in range(layers[l]):
        varname = "x_%d_%d"%(l, i)
        #print(varname)
        s = symbols(varname)
        #print(vars_x[i][j])
        vars_x[l].append(s)
        assert vars_x[l][i] == s


# x_li -> 

"""
x1 --> y1
xi --> yj  w_ij
xi --> x(i+1)_j
"""
vars_w = []
for l in range(len(layers)-1):
    #vars_w[l].append([])
    vars_w.append([])
    for i in range(layers[l]):
        #vars_w[l][i].append([])
        vars_w[l].append([])
        for j in range(layers[l+1]):
            # i -> j , l -> l+1
            parname = "w_%d_%d%d"%(l,i,j)
            s = symbols(parname)
            vars_w[l][i].append( s )
            assert vars_w[l][i][j] == s
    #vars_w[l] = []


for l in range(len(layers)-1):
    for i in range(layers[l]):
        for j in range(layers[l+1]):
            # i -> j , l -> l+1
            print (vars_w[l][i][j], end=", ")
        print()

vars_y = []
for l in range(len(layers)):
    vars_y.append([])
    for j in range(layers[l]):
        vars_y[l].append([])
        if l == 0:
            vars_y[l][j] = vars_x[0][j]
        else:
            x = 0
            for i in range(layers[l-1]):
                # i -> j, l -> l+1
                wname = "w_%d_%d%d"%(l, i, j)
                x = vars_y[l-1][j] * vars_w[l-1][i][j] + x
                #print(x)
            vars_y[l][j] = x
            print(x,"***")


#from typing import Mapping, Sequence, Callable, List
# Vector = List[List[]]

