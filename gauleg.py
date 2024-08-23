def gauleg(n):
	eps = 2.22044604925031e-16
	x = [ 0 for i in range(n) ]
	w = [ 0 for i in range(n) ]
	m = (n+1)/2
	xm = 0.0
	xl = 1.0
	i = 0
	while i <= (m-1):
		z = math.cos( math.pi*(i+1-0.25)/(n+0.5) )
		while 1:
			p1 = 1.0
			p2 = 0.0
			for j in range(1,n+1):
				p3 = p2
				p2 = p1
				p1 = ( (2.0*j-1.0)*z*p2 - (j-1.0)*p3)/j
			pp = n*( z*p1 - p2 )/(z*z - 1.0)
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