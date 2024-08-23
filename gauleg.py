
import math 
import numpy 

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
	return (2 + math.sin(x))

def GL(xi, ci, y, x, n2):
	sum = 0 
	for i in range(n2):
		prod = ci[i]*a(0.5*(((y-x)*xi[i])+(y+x)))
		sum = sum + prod 
	return (y-x)/2 * (sum)

def phiij(i,j):
	

	return (i,j)

def hat_integral(d_phi_i , d_phi_j, func, *args, **kwargs):
	sum = 0 
	i = 0 
	j = 0 
	for i in listij:
		for j in listij:
			for i in range(len(xlist)): 
				prod = GL(**kwargs) * d_phi_i * d_phi_j

			

	

