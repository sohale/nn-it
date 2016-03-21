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
layers = [3,2]


# prepare the space of variables
vars_x = [[None, None], [None,None], [None, None]] 
M = layers[0]
N = layers[1]
for i in range(M):
    for j in range(N):
        varname = "x_%d%d"%(i,j)
        #print(varname)
        vars_x[i][j] = symbols(varname)
        #print(vars_x[i][j])

#contains sympy variables
for i in range(M):
    for j in range(N):
        print(vars_x[i][j], end=", ")
    print()

from typing import Mapping, Sequence, Callable, List
Vector = List[List[]]
