from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
from numpy import diag, matrix, inf
from openopt import QP
import math

"""
https://github.com/smriti100jain/svr
"""

#------------------------------------------------------------------------------------------------------------
def kernel_value(x,y):
	a=math.exp(-1*abs(x-y)**2)
	return a

def product(a,X,x):
	prod=0.0
	for i in range(len(a)):
		prod=prod+a[i]*kernel_value(X[i],x)
	return prod
#-------------------------------------------------------------------------------------------------------------
eps=0.5
C=100

"""
X=[]
Y=[]

#taking input from the file
tot_values=int(raw_input())
for i in range(tot_values):
	X.append(float(raw_input()))
	Y.append(float(raw_input()))
"""
tot_values=350
mean=0
variance=0.5
lower_limit=0
upper_limit=10
Y=[]
X=[]
X.append(lower_limit)
for i in range(tot_values-1):
	X.append(X[-1]+float(upper_limit-lower_limit)/tot_values)
for i in X:
	Y.append(math.sin(float(i))+np.random.normal(mean,variance))
#-----------------------------------------------------------------------------------------------------------------


#H=kernel matrix
kernel=[[0.0 for i in range(2*tot_values)] for j in range(2*tot_values)]

for i in range(tot_values):
	for j in range(tot_values):
		kernel[i][j]=kernel_value(X[i],X[j])
		kernel[i+tot_values][j+tot_values]=kernel_value(X[i],X[j])

#----------------------------------------------------------------------------------------------------------------
#negating the values for a_n'
for i in range(tot_values):
	for j in range(tot_values):
		kernel[i+tot_values][j]=(-1.0)*kernel_value(X[i],X[j])
		kernel[i][j+tot_values]=(-1.0)*kernel_value(X[i],X[j])

#--------------------------------------------------------------------------------------------------------------
#coeff of 2nd term to minimize
f=[0.0 for i in range(2*tot_values)]
for i in range(tot_values):
	f[i]=-float(Y[i])+eps
for i in range(tot_values,2*tot_values):
	f[i]=float(Y[i-tot_values])+eps

#-----------------------------------------------------------------------------------------------------
#constraints
lower_limit=[0.0 for i in range(2*tot_values)]
upper_limit=[float(C) for i in range(2*tot_values)]
Aeq = [1.0 for i in range(2*tot_values)]
for i in range(tot_values,2*tot_values):
	Aeq[i]=-1.0
beq=0.0


#----------------------------------------------------------------------------------------------------

#coeff for 3rd constraint
#kernel=H
eq = QP(np.asmatrix(kernel),np.asmatrix(f),lb=np.asmatrix(lower_limit),ub=np.asmatrix(upper_limit),Aeq=Aeq,beq=beq)
p = eq._solve('cvxopt_qp', iprint = 0)
f_optimized, x = p.ff, p.xf

#---------------------------------------------------------------------------------------
support_vectors=[]
support_vectors_Y=[]
support_vector=[]
support_vector_Y=[]

coeff=[]
b=0.0
#support vectors: points such that an-an' ! = 0
for i in range(tot_values):
	if not((x[i]-x[tot_values+i])==0):
		support_vectors.append( X[i] )
		support_vectors_Y.append(Y[i])
		coeff.append( x[i]-x[tot_values+i] )
#lst = [237, 72, -18, 237, 236, 237, 60, -158, -273, -78, 492, 243]
#min((abs(x), x) for x in lst)[1]
#support vectors: points such that an-an' ! = 0
#Since some of the values of (x[i]-x[tot+i]) are very very close to zero and not zero
#support vectors are calculated as follows . if it is less tha 0.005 then it is cansidered to be a support vector
low=min(abs(x))
for i in range(tot_values):
	if not(abs(x[i]-x[tot_values+i])<low+0.005):
		support_vector.append( X[i] )
		support_vector_Y.append(Y[i])
		

#bias_term=tn-eps-(support vectors)*corresponding kernel
bias=0.0
for i in range(len(X)):
	bias=bias+float(Y[i]-eps-product(coeff,support_vectors,X[i]))
#generally bias is average as written in the book
bias=bias/len(X)


output_X=[]
output_Y=[]

output_X.append(0.0)

for i in range(350):
	output_X.append(output_X[-1]+float(10)/300)
out_eps=[]
out_eps1=[]
for i in output_X:
	output_Y.append(product(coeff,support_vectors,i)+b)
	out_eps.append(product(coeff,support_vectors,i)+b-eps)
	out_eps1.append(product(coeff,support_vectors,i)+b+eps)

plt.scatter(output_X,output_Y,c="red",marker='o')
plt.scatter(output_X,out_eps,marker='o')
plt.scatter(output_X,out_eps1,marker='o')

plt.scatter(X,Y,c="red",marker='.')
print(support_vector)
print(len(support_vectors))
plt.scatter(support_vector,support_vector_Y,c="green",marker='x')
print(len(support_vector))
plt.show()
print(low)
