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


#from typing import Mapping, Sequence, Callable, List
# Vector = List[List[]]


#layers = [3,4,2]
layers = [3, 2, 3]

#contains sympy variables
#for i in range(M):
#    for j in range(N):
#        print(vars_x[i][j], end=", ")
#    print()

# prepare the space of variables
#vars_x = [[None, None], [None,None], [None, None]] 
vars_x = []
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
        print(vars_x[l][i])
print()

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
vars_yx = []  # y based on immediate x
for l in range(len(layers)):
    vars_y.append([])
    vars_yx.append([])
    for j in range(layers[l]):
        vars_y[l].append("DUMMY")
        vars_yx[l].append("DUMMY")
        #assert exists vars_y[l][j] 
        print("len=", len(vars_yx[l]))
        if l == 0:
            vars_y[l][j] = vars_x[0][j]
            print("Layer %d, y(%d): "%(l,j), vars_y[l][j])
        else:
            s = 0
            for i in range(layers[l-1]):
                # i -> j, l -> l+1
                wname = "w_%d_%d%d"%(l, i, j)
                s = vars_y[l-1][i] * vars_w[l-1][i][j] + s
                #print(s)
            vars_y[l][j] = tanh(s)

            s = 0
            for i in range(layers[l-1]):
                # i -> j, l -> l+1
                wname = "w_%d_%d%d"%(l, i, j)
                s = vars_x[l-1][i] * vars_w[l-1][i][j] + s
            yx = tanh(s)
            vars_yx[l][j] = yx

            print("Layer %2d, y(%d): "%(l,j), vars_y[l][j])
            print("Layer %2d, x(%d): "%(l,j), vars_yx[l][j])
        print("len2=", len(vars_yx[l]))

print()
for l in range(len(layers)):
    for j in range(layers[l]):
        print(diff(vars_y[l][j],vars_x[0][0]))

print()
for l in range(1, len(layers)):
    for j in range(layers[l]):
        for i in range(layers[l-1]):
            by = vars_y[l-1][i]
            y = vars_y[l][j]
            print("d[%s]/d[%s]  = "% ("y","y"), #(str(y), str(by),), 
                diff(y, by))
            #simplify: too slow

print()
for l in range(len(layers)):
    for j in range(layers[l]):
        for i in range(layers[l-1]):
            print("y based on previous x: ", diff(vars_yx[l][j], vars_x[l-1][i]))



""" Generate C code """
from sympy.utilities.codegen import codegen


# sympy.utilities.codegen.codegen(name_expr, language, prefix=None, project='project', to_files=False, header=True, empty=True, argument_sequence=None, global_vars=None)

formula = diff(vars_y[l][j],vars_x[0][0])

#from sympy.abc import x, y, z
[(c_name, c_code), (h_name, c_header)] = codegen(  ("f1", formula), "C", "test", header=False, empty=False)

print(c_name)
print(c_code)


#for other languages see: http://docs.sympy.org/dev/_modules/sympy/utilities/codegen.html
# Also: https://github.com/jsyk/spnsyn-demo/blob/master/ftl/vhdlcodegen.py
#todo: quantum computing
