
import math 
import numpy 
import sympy 

def gauleg(n2):
	eps = 2.22044604925031e-16
	x = [ 0 for i in range(n2) ]
	w = [ 0 for i in range(n2) ]
	m = (n2+1)/2
	xm = 0.0
	xl = 1.0
	i = 0
	while i <= (m-1):
		z = math.cos( math.pi*(i+1-0.25)/(n2+0.5) )
		while 1:
			p1 = 1.0
			p2 = 0.0
			for j in range(1,n2+1):
				p3 = p2
				p2 = p1
				p1 = ( (2.0*j-1.0)*z*p2 - (j-1.0)*p3)/j
			pp = n2*( z*p1 - p2 )/(z*z - 1.0)
			z1 = z
			z = z1 - p1/pp
			if (abs(z-z1)<eps):
				break
		x[i] = xm - xl*z
		x[-i-1] = xm + xl*z
		w[i] = 2.0*xl/((1.0-z*z)*pp*pp)
		w[-i-1] = w[i]
		i = i + 1
	return x, w

def listi(a,b,h,n1):
	listi = []
	w = 0 
	x = a + w*h 
	while w < n1 :
		listi.append(x)
		w += 1 
		x = a + w*h 
	return listi 


def a(x):
	return 5*x +1

def GL(xi, ci, y, x, n2, func):
	sum = 0 
	for i in range(n2):
		prod = ci[i]*func(0.5*(((y-x)*xi[i])+(y+x)))
		sum = sum + prod 
	return (y-x)/2 * (sum)

def phiij(numofnodes, i, j, l, xlist):
	finalsum = 0
	for k in range(numofnodes-1):

		x = xlist[k]
		y = xlist[k+1]
	 
		if k == i :
			d_phi_i = 2**l 
		elif k == i + 1 :
			d_phi_i = -2**l 
		else: 
			d_phi_i = 0 
 
		if k == j : 
			d_phi_j = 2**l 
		elif k == j + 1 :
			d_phi_j = -2**l 
		else:
			d_phi_j = 0 

		n2 = 5 
		xi = gauleg(n2)[0]
		ci = gauleg(n2)[1]
		onegl = GL(xi = xi, ci = ci, y = y, x = x, n2 = 5, func = a) * d_phi_i *d_phi_j
		finalsum = finalsum + onegl

	return finalsum



def matrix(ijlist, matrix, n1, l, xlist):
	for i in ijlist:
		for j in ijlist:
			aij = phiij(numofnodes=n1, i=i, j =j, l = l, xlist=xlist)
			matrix[i,j] = aij 

	return matrix 



def f(s):
    return math.sin(s)

def intergrand():
    def phi_j(numofnodes, i, l):
        def phi_j_function(s):
            for k in range(numofnodes-1):    
                
                if k == i:
                    phi_j_value = 2**l * (s - (i / 2**l))
                elif k == i + 1:
                    phi_j_value = -(2**l * (s - ((i + 2) / 2**l)))
                else: 
                    phi_j_value = 0

                h = f(s) * phi_j_value

            return h
        return phi_j_function
    return phi_j

def GLF_sum(func, xlist, n2, numofnodes):
	finalsum = 0
	xi = gauleg(n2)[0]
	ci = gauleg(n2)[1]
	for k in range(numofnodes-1):
		x = xlist[k]
		y = xlist[k+1]
		onegl = GL(xi = xi, ci = ci, y =y, x = x, n2 = 5, func = func )
		finalsum = finalsum + onegl
	return finalsum 


