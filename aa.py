

p_hat = 5. / 8.
freq_prob = (1 - p_hat) ** 3
print(freq_prob)

#########################################

print("Odds against Bob winning: {0:.0f} to 1".format((1. - freq_prob) / freq_prob))

#########################################

from scipy.special import beta
bayes_prob = beta(6 + 1, 5 + 1) / beta(3 + 1, 5 + 1)

print("P(B|D) = {0:.2f}".format(bayes_prob))

#########################################

print("Bayesian odds against Bob winning: {0:.0f} to 1".format((1. - bayes_prob) / bayes_prob))

#########################################

import numpy as np
np.random.seed(0)

# play 100000 games with randomly-drawn p, between 0 and 1
p = np.random.random(100000)

# each game needs at most 11 rolls for one player to reach 6 wins
rolls = np.random.random((11, len(p)))

# count the cumulative wins for Alice and Bob at each roll
Alice_count = np.cumsum(rolls < p, 0)
Bob_count = np.cumsum(rolls >= p, 0)

# sanity check: total number of wins should equal number of rolls
total_wins = Alice_count + Bob_count
assert np.all(total_wins.T == np.arange(1, 12))
print("(Sanity check passed)")

# determine number of games which meet our criterion of (A wins, B wins)=(5, 3)
# this means Bob's win count at eight rolls must equal 3
good_games = Bob_count[7] == 3
print("Number of suitable games: {0}".format(good_games.sum()))

# truncate our results to consider only these games
Alice_count = Alice_count[:, good_games]
Bob_count = Bob_count[:, good_games]

# determine which of these games Bob won.
# to win, he must reach six wins after 11 rolls.
bob_won = np.sum(Bob_count[10] == 6)
print("Number of these games Bob won: {0}".format(bob_won.sum()))

# compute the probability
mc_prob = bob_won.sum() * 1. / good_games.sum()
print("Monte Carlo Probability of Bob winning: {0:.2f}".format(mc_prob))
print("MC Odds against Bob winning: {0:.0f} to 1".format((1. - mc_prob) / mc_prob))


# Number of suitable games: 11068
# Number of these games Bob won: 979
# Monte Carlo Probability of Bob winning: 0.09
# MC Odds against Bob winning: 10 to 1


###############################################

x = np.array([ 0,  3,  9, 14, 15, 19, 20, 21, 30, 35,
              40, 41, 42, 43, 54, 56, 67, 69, 72, 88])
y = np.array([33, 68, 34, 34, 37, 71, 37, 44, 48, 49,
              53, 49, 50, 48, 56, 60, 61, 63, 44, 71])
e = np.array([ 3.6, 3.9, 2.6, 3.4, 3.8, 3.8, 2.2, 2.1, 2.3, 3.8,
               2.2, 2.8, 3.9, 3.1, 3.4, 2.6, 3.4, 3.7, 2.0, 3.5])

###############################################

import matplotlib.pyplot as plt

plt.errorbar(x, y, e, fmt='.k', ecolor='gray');

###############################################

from scipy import optimize

def squared_loss(theta, x=x, y=y, e=e):
    dy = y - theta[0] - theta[1] * x
    return np.sum(0.5 * (dy / e) ** 2)

theta1 = optimize.fmin(squared_loss, [0, 0], disp=False)

xfit = np.linspace(0, 100)
plt.errorbar(x, y, e, fmt='.k', ecolor='gray')
plt.plot(xfit, theta1[0] + theta1[1] * xfit, '-k')
plt.title('Maximum Likelihood fit: Squared Loss');

###############################################

t = np.linspace(-20, 20)

def huber_loss(t, c=3):
    return ((abs(t) < c) * 0.5 * t ** 2
            + (abs(t) >= c) * -c * (0.5 * c - abs(t)))

plt.plot(t, 0.5 * t ** 2, label="squared loss", lw=2)
for c in (10, 5, 3):
    plt.plot(t, huber_loss(t, c), label="Huber loss, c={0}".format(c), lw=2)
plt.ylabel('loss')
plt.xlabel('standard deviations')
plt.legend(loc='best', frameon=False);

###############################################

def total_huber_loss(theta, x=x, y=y, e=e, c=3):
    return huber_loss((y - theta[0] - theta[1] * x) / e, c).sum()

theta2 = optimize.fmin(total_huber_loss, [0, 0], disp=False)

plt.errorbar(x, y, e, fmt='.k', ecolor='gray')
plt.plot(xfit, theta1[0] + theta1[1] * xfit, color='lightgray')
plt.plot(xfit, theta2[0] + theta2[1] * xfit, color='black')
plt.title('Maximum Likelihood fit: Huber loss');

################################################

# theta will be an array of length 2 + N, where N is the number of points
# theta[0] is the intercept, theta[1] is the slope,
# and theta[2 + i] is the weight g_i

def log_prior(theta):
    #g_i needs to be between 0 and 1
    if (all(theta[2:] > 0) and all(theta[2:] < 1)):
        return 0
    else:
        return -np.inf  # recall log(0) = -inf

def log_likelihood(theta, x, y, e, sigma_B):
    # print(theta[0])
    # print(theta[1])
    # print("____________________________________________________")
    dy = y - theta[0] - theta[1] * x
    g = np.clip(theta[2:], 0, 1)  # g<0 or g>1 leads to NaNs in logarithm
    logL1 = np.log(g) - 0.5 * np.log(2 * np.pi * e ** 2) - 0.5 * (dy / e) ** 2
    logL2 = np.log(1 - g) - 0.5 * np.log(2 * np.pi * sigma_B ** 2) - 0.5 * (dy / sigma_B) ** 2
    return np.sum(np.logaddexp(logL1, logL2))

def log_posterior(theta):
    x1 = np.array([ 0,  3,  9, 14, 15, 19, 20, 21, 30, 35,
                  40, 41, 42, 43, 54, 56, 67, 69, 72, 88])
    y1 = np.array([33, 68, 34, 34, 37, 71, 37, 44, 48, 49,
                  53, 49, 50, 48, 56, 60, 61, 63, 44, 71])
    e1 = np.array([ 3.6, 3.9, 2.6, 3.4, 3.8, 3.8, 2.2, 2.1, 2.3, 3.8,
                   2.2, 2.8, 3.9, 3.1, 3.4, 2.6, 3.4, 3.7, 2.0, 3.5])
    sigma_B = 50
    # theta = theta.T
    theta = np.squeeze(np.asarray(theta))
    print(theta)
    print("________________________________________________________________________")
    return log_prior(theta) + log_likelihood(theta, x, y, e, sigma_B)


def log_posterior1(theta,x,y,e,sigma_B):
    # x1 = np.array([ 0,  3,  9, 14, 15, 19, 20, 21, 30, 35,
    #               40, 41, 42, 43, 54, 56, 67, 69, 72, 88])
    # y1 = np.array([33, 68, 34, 34, 37, 71, 37, 44, 48, 49,
    #               53, 49, 50, 48, 56, 60, 61, 63, 44, 71])
    # e1 = np.array([ 3.6, 3.9, 2.6, 3.4, 3.8, 3.8, 2.2, 2.1, 2.3, 3.8,
    #                2.2, 2.8, 3.9, 3.1, 3.4, 2.6, 3.4, 3.7, 2.0, 3.5])
    # sigma_B = 50
    # theta = theta.T
    # print(theta)
    # print("________________________________________________________________________")
    return log_prior(theta) + log_likelihood(theta, x, y, e, sigma_B)

#################################################

# Note that this step will take a few minutes to run!

ndim = 2 + len(x)  # number of parameters in the model
nwalkers = 50  # number of MCMC walkers
nburn = 10000  # "burn-in" period to let chains stabilize
nsteps = 15000  # number of MCMC steps to take

# set theta near the maximum likelihood, with
np.random.seed(0)
starting_guesses = np.zeros((nwalkers, ndim))
starting_guesses[:, :2] = np.random.normal(theta1, 1, (nwalkers, 2))
starting_guesses[:, 2:] = np.random.normal(0.5, 0.1, (nwalkers, ndim - 2))
# #
print(len(starting_guesses[49]))

import emcee
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior1, args=[x, y, e, 50])
sampler.run_mcmc(starting_guesses, nsteps)

sample = sampler.chain  # shape = (nwalkers, nsteps, ndim)
sample = sampler.chain[:, nburn:, :].reshape(-1, ndim)


##########################
#             Bayesian part

import GPy
import GPyOpt
from GPyOpt.methods import BayesianOptimization
# # func = log_posterior(starting_guesses[1], x, y, e, 50)
#
domain = [{'name': 'var_1', 'type': 'continuous', 'domain': (-1,1), 'dimensionality':22 }]   #theta dim = 22

myBopt = BayesianOptimization(f=log_posterior, domain = domain, initial_design_numdata = 10, acquisition_type = 'EI', exact_feval = 'False')
myBopt.run_optimization(max_iter = 30, eps = 0)
print(np.mean(myBopt.x_opt))
# print(myBopt.fx_opt)

#
# np.round(bo.x_opt,2)
#
# np.round(func.min[0],2)

##################################################

plt.plot(sample[:, 0], sample[:, 1], ',k', alpha=0.1)
plt.xlabel('intercept')
plt.ylabel('slope');

##################################################

plt.plot(sample[:, 2], sample[:, 3], ',k', alpha=0.1)
plt.xlabel('$g_1$')
plt.ylabel('$g_2$')

print("g1 mean: ".format(sample[:, 2]))
print("g2 mean: ".format(sample[:, 3]))

##################################################

theta3 = np.mean(sample[:, :2], 0)
g = np.mean(sample[:, 2:], 0)
g1 = np.mean
outliers = (g < 0.5)
print(outliers)
plt.errorbar(x, y, e, fmt='.k', ecolor='gray')
# plt.plot(xfit, theta1[0] + theta1[1] * xfit, color='lightgray')
# plt.plot(xfit, theta2[0] + theta2[1] * xfit, color='lightgray')
# plt.plot(xfit, theta3[0] + theta3[1] * xfit, color='black')


# plt.plot(x[outliers], y[ m,liers], 'ro', ms=20, mfc='none', mec='red')
plt.title('Maximum Likelihood fit: Bayesian Marginalization');
plt.show()


##################################################
