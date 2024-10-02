import math
import sympy as sp 

def Eulersys(t,h,y10,y20,t0):
    n=math.floor((t-t0)/h)
    y1=y10 # initial value of y 
    y2=y20 # inital value of y'
    for i in range(n):
        newy1=y1+h*f1(t0+h*i,y1,y2)
        newy2=y2+h*f2(t0+h*i,y1,y2)
        y1=newy1
        y2=newy2
    return [y1,y2]

def f1(t,x1, y1):
    value_f1=y1
    return value_f1

def f2(t,x1,y1):
    value_f2=-((2-t)/t)*x1 +(2/t)*y1
    return value_f2

def approx():
    c_0, c_1 = sp.symbols('c_0, c_1')
    eq1 = sp.Eq(math.e * c_0  + 5 * c_1* math.e ** (-1), 3) 
    eq2 = sp.Eq(math.e * c_0 + (-2 * math.e ** (-1) + 2 * math.e **(-1) + math.e **(-1))* c_1, 4)
    result = sp.solve([eq1, eq2], (c_0 , c_1))
    return result
c_0, c_1 = sp.symbols('c_0, c_1')

def actual(t):
    a = actual 
    a = approx()[c_0] * math.e **(t) + (1 + 2* t + 2*t**2) * math.e ** (-t) * approx()[c_1]
    return a 
reason = Eulersys(10, 0.00000001, 3, 4, 1)
print(actual(10) - reason[0])



