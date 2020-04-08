import numpy
import matplotlib.pyplot as pyplot
import matplotlib.gridspec as gridspec

mu = -2
t = 0
U = 0.01

Y, X = numpy.mgrid[-0.1:0.1:200j, -20:0:200j]
U = Y
V = -mu*X + t**2*X + U*X**3

pyplot.streamplot(X,Y,U,V,density=1)
pyplot.show()