import numpy as np
import scipy.stats as sp
import scipy
import matplotlib.pyplot as plt
# EXAMPLE 2


# x - Coordinates
x = np.array([ 0,  3,  9, 14, 15, 19, 20, 21, 30, 35, 40, 41, 42, 43, 54, 56, 67, 69, 72, 88])    # 20 points 
# y - Coordinates
y = np.array([33, 68, 34, 34, 37, 71, 37, 44, 48, 49, 53, 49, 50, 48, 56, 60, 61, 63, 44, 71])

# errors of output values for each x
e = np.array([ 3.6, 3.9, 2.6, 3.4, 3.8, 3.8, 2.2, 2.1, 2.3, 3.8, 2.2, 2.8, 3.9, 3.1, 3.4, 2.6, 3.4, 3.7, 2.0, 3.5])



# the posterior in this code was originally optimized by using Monte-Carlo-Markov-Chain optimizer




from scipy import optimize

def squared_loss(theta, x=x, y=y, e=e):
    dy = y - theta[0] - theta[1] * x
    return np.sum(0.5 * (dy / e) ** 2)

theta1 = optimize.fmin(squared_loss, [0, 0], disp=False)

xfit = np.linspace(0, 100)
tempY = [0]


# FIND THE BEST FIT LINE FOR THIS DATA 
# THERE ARE SOME POINTS THAT OUTLIE THE GIVEN DATA 
# y = a0 + a1 * x
# since we have the points we can compute the gaussian likelihoods
# we need to maximize this log likelihood
# squared loss efficiency not optimum maybe because the parameter we have doesnt lead to the global minima 
 
## we ignore the huber loss function approach

## now we choose a mixture between signal and noise

## we now increased the dimensionality of our model
## most of those parameter are nuisance parameters which are marginalised at the end
## We now make functions to describe our prior and posterior(likelihood) functions and out posterior

# out final function is a sum of prior and posterior 
# we take logs on both sides and find the probabilities 




##############################################################################################################################
##             OUR LOGARITHMIC POSTERIOR FUNCTION, we need to find the parameters which maximizes this function

# actually we need only two parameters, but we extended the number of parameters to 22 hoping the redundant parameters get marginalized over the end
# Q : why was there a need for that?

					####################################################################################
# theta will be an array of length 2 + N, where N is the number of points
# theta[0] is the intercept, theta[1] is the slope,
# and theta[2 + i] is the weight g_i

slopes = [0]
def log_prior(theta):
	# g_i is between 0,1
	# theta[2+i] is the weight g_i
	#temp = theta[:,:]
	#temp1 = temp[0]
	if(all(theta[2:] > 0) and all(theta[2:] < 1)):
		return 0
	else:
		return -np.inf # since log(0) = -inf

def log_likelihood(theta,x,y,e,sigma_B):   # this function working like it is marginalizing all the other parameters
	c = theta[0]
	m = theta[1]	
	dy = y - c - m*x
	g = np.clip(theta[2:],0,1)        # has dimension 20                                                        # PROBLEM, How is this 'g' getting appended
	print(len(g))	
	logL1 = np.log(g) - 0.5*np.log(2*np.pi* e**2) - 0.5*(dy/e)**2                                                 # 'g' is using the nuisance parameters to get updated. BUT WHY?
	logL2 = np.log(1-g) - 0.5*np.log(2*np.pi*sigma_B**2) - 0.5*(dy/sigma_B)**2
	return np.sum(np.logaddexp(logL1, logL2))    # should be constant value                                 # logaddexp() is a special function in python specially for statistics 

def log_posterior(theta):
	theta = np.asarray(theta)
	# x - Coordinates
	x = np.array([ 0,  3,  9, 14, 15, 19, 20, 21 ])    # 20 points 
	# y - Coordinates
	y = np.array([33, 68, 34, 34, 37, 71, 37, 44])
	# errors of output values for each x
	e = np.array([ 3.6, 3.9, 2.6, 3.4, 3.8, 3.8, 2.2, 2.1])	
	print( "slopes = ",( log_likelihood(theta,x,y,e,50)))
	
	slopes.append(theta[1])	
	
	return log_likelihood(theta,x,y,e,50)

################################################################################################################################
## estimation by Bayesian methods


theta =np.ones(10)
for i in range (1,20):
	theta = np.vstack((theta,np.ones(10))) 
	theta[i,0] = i-10
	theta[i,1] = (i-10)

yX = [0]
yP = np.linspace(-10,10,18)

print (theta) 

for i in range (0,19):
	tempY = log_posterior(theta[i])
	yX.append(tempY)

print(yX)
yX = yX[2:]
print(len(theta[:,1]))
plt.ylabel('posterior')
plt.xlabel('intercept')
plt.plot(yP,yX)
plt.show()








