# -*- coding: utf-8 -*-
"""
Created on Sun Jan  7 14:02:00 2018

@author: Harsh
@Notes:
--> This file consists of all kinds of gradient descent algorithms in the
 pooled case
--> For all the algorithm, a tolerance of 1e-5/1e-6 is assumed consistently
-->
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)


def objective(w, X, y, lamb=0.0):
    return (1 / 2 * len(X)) * np.sum(
        (np.dot(X, w) - y)**2) + lamb * np.dot(w.T, w) / 2.


def gradient(w, X, y, lamb=0.0):
    return (1 / len(X)) * np.dot(X.T, np.dot(X, w) - y) + lamb * w


def gottol(v, tol=1e-5):
    return np.sum(np.square(v)) <= tol


# Number of steps is critical
# As eta goes down number of steps should go up
def gd_for(winit, X, y, steps=1000, eta=0.01, tol=1e-6):
    wc = winit

    for i in range(steps):
        Gradient = gradient(wc, X, y)
        wc = wc - eta * gradient(wc, X, y)
        if gottol(Gradient, tol):
            break
    return wc, i


# Number of steps doesnt play a role at all
# Tolerance determines the final output
# Probably a high learning rate might be dangerous
def gd_while(winit, X, y, steps=100, eta=0.01, tol=1e-6):
    wc = winit

    count = 0
    Gradient = gradient(wc, X, y)
    while not gottol(Gradient, tol):
        count = count + 1
        wc = wc - eta * Gradient
        Gradient = gradient(wc, X, y)
    return wc, count


# Written by Sergey
# Address the danger of having a high initial learning rate
# Number of steps need to be high (because it's a for loop)
def gd_for_heuristic(winit,
                     X,
                     y,
                     grad=gradient,
                     obj=objective,
                     steps=1000,
                     eta=0.01,
                     tol=1e-6):
    wc = wp = winit
    pObj = obj(wc, X, y)
    co = []

    for i in range(steps):
        cObj = objective(wc, X, y)
        if cObj > pObj:
            eta = eta / 2
            continue

        Gradient = gradient(wc, X, y)
        if gottol(Gradient, tol):
            break
        wc = wp - eta * Gradient
        co.append(cObj)

        pObj = cObj
        wp = wc
    return wc, co, i


# Written by Harsh
# A modification of gd_for_heuristic()
# The for loop is replaced by the while loop
# Number of steps is irrelevant
def gd_while_heuristic(winit,
                       X,
                       y,
                       grad=gradient,
                       obj=objective,
                       steps=1000,
                       eta=0.01,
                       tol=1e-6):
    wc = wp = winit
    pObj = obj(wc, X, y)
    co = []

    count = 0
    Gradient = gradient(wc, X, y)
    while not gottol(Gradient, tol):
        count = count + 1
        wc = wp - eta * Gradient
        cObj = objective(wc, X, y)
        if cObj > pObj:
            eta = eta / 2
            continue
        else:
            co.append(cObj)
            pObj = cObj
            wp = wc
            Gradient = gradient(wc, X, y)
    return wc, co, count


# Written by Sergey
# The Number of Steps seems to hold the key (probably because its a for loop)
def adoptimizer(winit,
                X,
                y,
                grad=gradient,
                obj=objective,
                steps=50000,
                rho=0.99,
                eps=1e-4,
                tol=1e-5):
    wc = winit
    pObj = obj(wc, X, y)
    Eg2 = 0
    EdW2 = 0
    co = []

    for i in range(steps):
        cObj = objective(wc, X, y)
        if cObj > pObj:
            break

        Gradient = gradient(wc, X, y)
        if gottol(Gradient, tol):
            break

        # ADADELTA
        Eg2 = rho * Eg2 + (1 - rho) * Gradient * Gradient
        dW = -np.sqrt(EdW2 + eps) / np.sqrt(Eg2 + eps) * Gradient
        EdW2 = rho * EdW2 + (1 - rho) * dW * dW
        wc = wc + dW
        co.append(cObj)

        pObj = cObj

    return wc, co, i


# Written by Harsh
# A modification of adoptimizer
# Replaced the for loop with while loop
def adoptimizer_while(winit,
                      X,
                      y,
                      grad=gradient,
                      obj=objective,
                      steps=50000,
                      rho=0.99,
                      eps=1e-4,
                      tol=1e-5):
    wc = winit
    pObj = obj(wc, X, y)
    Eg2 = 0
    EdW2 = 0
    co = []

    count = 0
    Gradient = gradient(wc, X, y)
    while not gottol(Gradient, tol):
        count = count + 1
        cObj = objective(wc, X, y)
        if cObj > pObj:
            break

        Gradient = gradient(wc, X, y)

        # ADADELTA
        Eg2 = rho * Eg2 + (1 - rho) * Gradient * Gradient
        dW = -np.sqrt(EdW2 + eps) / np.sqrt(Eg2 + eps) * Gradient
        EdW2 = rho * EdW2 + (1 - rho) * dW * dW
        wc = wc + dW
        co.append(cObj)

        pObj = cObj

    return wc, co, count


# Written by Sergey (backtracking line search)
def btoptimizer(winit,
                X,
                y,
                grad=gradient,
                obj=objective,
                steps=10000,
                eta=0.01,
                mult=0.5,
                tol=1e-2):

    wc = wp = winit
    gradval = grad(wc, X, y)
    pObj = obj(wc, X, y)
    backtrack = False
    co = []

    for i in range(steps):
        wc = wp - eta * gradval  # we assume the gradient exists and make a step
        cObj = objective(wc, X, y)  # check the new objective

        if cObj > pObj:
            eta *= mult
            backtrack = True

        if not backtrack:
            gradval = gradient(wc, X, y)
            wp = wc
            co.append(cObj)
            pObj = cObj

        if gottol(gradval, tol):
            break
        backtrack = False

    return wc, co, i


# Written by Harsh
# A modification of btoptimizer() function
# replaces the for loop with a while loop
def btoptimizer_while(winit,
                      X,
                      y,
                      grad=gradient,
                      obj=objective,
                      steps=10000,
                      eta=0.01,
                      mult=0.5,
                      tol=1e-2):

    wc = wp = winit
    gradval = grad(wc, X, y)
    pObj = obj(wc, X, y)
    backtrack = False
    co = []

    count = 0
    while not gottol(gradval, tol):
        count = count + 1
        wc = wp - eta * gradval  # we assume the gradient exists and make a step
        cObj = objective(wc, X, y)  # check the new objective

        if cObj > pObj:
            eta *= mult
            backtrack = True

        if not backtrack:
            gradval = gradient(wc, X, y)
            wp = wc
            co.append(cObj)
            pObj = cObj

        backtrack = False

    return wc, co, count


# backtracking line search
# =============================================================================
# def goptimizerbt(winit, X, y, steps=100, eta=0.01, tol=1e-6):
#     wc = winit
#     eta = 1.0
#     beta = 0.5
#     alpha = 0.0001
#     Gradient = gradient(wc, X, y, l=0.)
#     while not gottol(Gradient, tol):
#         while objective(wc + eta * Gradient, X, y) > objective(wc, X, y) + alpha * eta * Gradient:
#             eta = eta * beta
#         wc = wc - eta * Gradient
#         Gradient = gradient(wc, X, y, l=0.)
#     return wc
# =============================================================================


# Written by Harsh
# Implements the momentum gradient descent
# Smaller the initial learning rate larger the number of steps
def gd_momentum(winit, X, y, steps=100, eta=0.01, tol=1e-6):
    vel_prev = wc = winit
    gamma = 0.9
    count = 0
    Gradient = gradient(wc, X, y)
    while not gottol(Gradient, tol):
        count = count + 1
        vel = gamma * vel_prev + eta * gradient(wc, X, y)
        wc = wc - vel
        vel_prev = vel
        Gradient = gradient(wc, X, y)
    return wc, count


# Written by Harsh
# Implements the Nesterov accelerated gradient descent
# Starting Learning Rate seems to be important
# Works well for eta = 1e-3 but gives error for 1e-2
def gd_nesterov(winit, X, y, steps=100, eta=0.01, tol=1e-6):
    vel_prev = wc = winit
    gamma = 0.9
    count = 0
    Gradient = gradient(wc, X, y)
    while not gottol(Gradient, tol):
        count = count + 1
        vel = gamma * vel_prev + eta * gradient(wc - gamma * vel_prev, X, y)
        wc = wc - vel
        vel_prev = vel
        Gradient = gradient(wc, X, y)
    return wc, count


# Implemented by Harsh
# https://github.com/benbo/adagrad
def adagrad(w, X, y, steps=1000, eta=1e-2, tol=1e-6):
    count = 0
    eps = 1e-8
    gti = np.zeros(w.shape[0])
    wc = w
    grad = gradient(wc, X, y)
    while not gottol(grad, tol):
        count = count + 1
        gti += grad**2
        adjusted_grad = grad / (eps + np.sqrt(gti))
        wc = wc - eta * adjusted_grad
        grad = gradient(wc, X, y)
    return wc, count


# Little Dangerous
def rmsprop(w, X, y, steps=0, eta=1e-3, tol=1e-6):
    wc = w
    eps = 1e-8
    rho = 0.9

    grad = gradient(wc, X, y)
    Eg2 = grad**2
    count = 0
    for i in range(steps):
        Eg2 = rho * Eg2 + (1 - rho) * (grad**2)
        adjusted_grad = grad / (eps + np.sqrt(Eg2))
        wc = wc - eta * adjusted_grad
        grad = gradient(wc, X, y)
        if gottol(grad, tol):
            break
    return wc, count


def adam(w, X, y, steps=10000, eta=1e-2, tol=1e-2):
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    wp = w
    mt, vt = 0, 0

    i = 0
    while True:
        i = i + 1
        grad = gradient(wp, X, y)

        mt = beta1 * mt + (1 - beta1) * grad
        vt = beta2 * vt + (1 - beta2) * (grad**2)

        m = mt / (1 - beta1**i)
        v = vt / (1 - beta2**i)

        wc = wp - eta * m / (np.sqrt(v) + eps)
        print(wc)
        if np.linalg.norm(wp - wc) <= tol:
            break

        wp = wc

    return wc, i


# https://onlinecourses.science.psu.edu/stat462/node/101
#df = pd.read_excel('test.xlsx')
#X = np.array(df['X'])
#X = np.array([[1] * len(X), X]).T
#y = np.array(df['Y']).T

#### msr_test_simple_example
#X = np.array([[1, 1], [1, 2], [1, 3], [1, 4], [1, 5]])
#y = np.array([1,2,3,4,5]).T
#w = np.array([0, 0])

# msr_test_fsl
X = np.array([[1,0,22], [1, 1, 47], [1, 0, 56], [1, 1, 73]])
#y = np.array([2115.49, 4019.55, 4934.63, 9360.22])  # 3rd ventricle
#y = np.array([1359.72, 3587.1, 3683.96, 8444.98])  # 4th ventricle
y = np.array([ 6188.1, 10824.89, 12807.25, 18943.92])  # 5th ventricle
w = np.array([0, 0, 0])

# Pooled Algorithms
#normal_w = np.matmul(np.linalg.inv(np.matmul(X.T, X)), np.matmul(X.T, y))
#print('{:<25}{}'.format('Normal Equation:', normal_w))
#
#model = sm.OLS(y, X.astype(float)).fit()
#print('{:<25}{}'.format('Stats Models:', model.params))
#
#w1, count1 = gd_for(w, X, y, steps=100000, eta=1e-3, tol=1e-6)
#print('{:<25}{}{}'.format('Vanilla_GD (for loop):', w1, count1))
#
#w2, count2 = gd_while(w, X, y, steps=100000, eta=1e-3, tol=1e-6)
#print('{:<25}{}{}'.format('Vanilla_GD (while loop):', w2, count2))
#
#w_v, c_v, count_v = gd_for_heuristic(w, X, y, steps=100000, eta=1e-3, tol=1e-6)
#print('{:<25}{}{}'.format('gd_for_heuristic:', w_v, count_v))
#
#w3, c3, count3 = gd_while_heuristic(w, X, y, steps=100000, eta=1e-3, tol=1e-6)
#print('{:<25}{}{}'.format('gd_while_heuristic:', w3, count3))
#
#w_a, c_a, count_a = adoptimizer(w, X, y, steps=100000, tol=1e-6)
#print('{:<25}{}{}'.format('adadelta (for loop):', w_a, count_a))
#
#w_a1, c_a1, count_a1 = adoptimizer_while(w, X, y, steps=100000, tol=1e-6)
#print('{:<25}{}{}'.format('adadelta (while loop):', w_a1, count_a1))
#
#w_b, c_b, count_b = btoptimizer(w, X, y, steps=100000, eta=1e-3, tol=1e-6)
#print('{:<25}{}{}'.format('backtracking (for):', w_b, count_b))
#
#w_b1, c_b1, count_b1 = btoptimizer_while(
#    w, X, y, steps=100000, eta=1e-3, tol=1e-6)
#print('{:<25}{}{}'.format('backtracking (while):', w_b1, count_b1))
#
#w_m, count_m = gd_momentum(w, X, y, steps=0, eta=1e-2, tol=1e-6)
#print('{:<25}{}{}'.format('Momentum:', w_m, count_m))
#
#w_n, count_n = gd_nesterov(w, X, y, steps=0, eta=1e-3, tol=1e-6)
#print('{:<25}{}{}'.format('Nesterov:', w_n, count_n))
#
#w_adg, count_adg = adagrad(w, X, y, steps=0, eta=1, tol=1e-6)
#print('{:<25}{}{}'.format('Adagrad:', w_adg, count_adg))
#
#w_rms, count_rms = rmsprop(w, X, y, steps=10000, eta=1e-2, tol=1e-6)
#print('{:<25}{}{}'.format('RMSProp:', w_rms, count_rms))

w_adam, count_adam = adam(w, X, y, steps=10000, eta=1000, tol=100)
print('{:<25}{}{}'.format('Adam:', w_adam, count_adam))
