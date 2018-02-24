# -*- coding: utf-8 -*-
"""
Created on Sun Jan  7 14:02:00 2018

@author: Harsh
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

def objective(w, X, y, l=0.0):
    return (1/2*len(X)) * np.sum((np.dot(X,w) - y)**2) + l * np.dot(w.T,w)/2.


def gradient(w, X, y, l=0.0):
    return (1/len(X)) * np.dot(X.T, np.dot(X,w) - y) + l * w


def gottol(v, tol=1e-1):
    return np.sum(np.square(v)) <= tol


# Number of steps is critical
# didnt give importance to the tolerance yet
# eta is also critical here
# as eta goes down number of steps should go up
# tolerance doesnt play a role here at all
def goptimizer1(winit, X, y, steps=1000, eta=0.01, tol=1e-6):
    wc = winit

    for i in range(steps):
        wc = wc - eta * gradient(wc, X, y)
    return wc

# In this tolerance determines the final output
#  and number of steps doesnt play a role at all
def goptimizer2(winit, X, y, steps=100, eta=0.01, tol=1e-6):
    wc = winit

    Gradient = gradient(wc, X, y, l=0.)
    while not gottol(Gradient, tol):
        wc = wc - eta * Gradient
        Gradient = gradient(wc, X, y, l=0.)
    return wc

# written by sergey
# number of steps need to be high
# made some minor modifications
def goptimizer(winit, X, y, grad=gradient, obj=objective, steps=1000, eta=0.01, tol=1e-6):
    wc = wp = winit
    pObj = obj(wc, X, y)
    co = []

    for i in range(steps):
        cObj = objective(wc, X, y, l=0.)
        if cObj > pObj:
            eta = eta/2
            continue

        Gradient = gradient(wc, X, y, l=0.)
        if gottol(Gradient, tol): break
        wc = wc - eta * Gradient
        co.append(cObj)

        pObj = cObj
        wp = wc
    return wc, co


def goptimizer3(winit, X, y, grad=gradient, obj=objective, steps=1000, eta=0.01, tol=1e-6):
    wc = wp = winit
    pObj = obj(wc, X, y)
    co = []

    Gradient = gradient(wc, X, y, l=0.)
    while not gottol(Gradient, tol):
        wc = wp - eta * Gradient
        cObj = objective(wc, X, y, l=0.)
        if cObj > pObj:
            eta = eta/2
            continue
        else:
            co.append(cObj)
            pObj = cObj
            wp = wc
            Gradient = gradient(wc, X, y, l=0.)
    return wc, co


# number of steps is key here as well
# written by Sergey
def adoptimizer(winit, X, y, grad=gradient, obj=objective, steps=50000,
                rho=0.99, eps=1e-4, tol=1e-5):
    wc = wp = winit
    pObj = obj(wc, X, y)
    Eg2 = 0
    EdW2 = 0
    co = []

    for i in range(steps):
        cObj = objective(wc, X, y, l=0.)
        if cObj > pObj: break

        Gradient = gradient(wc, X, y, l=0.)
        if gottol(Gradient, tol): break

        # ADADELTA
        Eg2 = rho * Eg2 + (1 - rho) * Gradient * Gradient
        dW = - np.sqrt(EdW2 + eps) / np.sqrt(Eg2 + eps) * Gradient
        EdW2 = rho * EdW2 + (1 - rho) * dW * dW
        wc = wc + dW
        co.append(cObj)

        pObj = cObj

    return wc, co

# By Sergey as well
def btoptimizer(winit, X, y, grad=gradient, obj=objective, steps=10000,
                eta=0.01, mult=0.5, tol=1e-2):

    wc = wp = winit
    gradval = grad(wc, X, y, l=0.)
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
            gradval = gradient(wc, X, y, l=0.)
            wp = wc
            co.append(cObj)
            pObj = cObj

        if gottol(gradval, tol):
            break
        backtrack = False

    return wc, co


def goptimizerwithbt(winit, X, y, steps=100, eta=0.01, tol=1e-6):
    wc = winit
    eta = 1.0
    beta = 0.5
    alpha = 0.0001
    Gradient = gradient(wc, X, y, l=0.)
    while not gottol(Gradient, tol):
        while objective(wc + eta * Gradient, X, y) > objective(wc, X, y) + alpha * eta * Gradient:
            eta = eta * beta
        wc = wc - eta * Gradient
        Gradient = gradient(wc, X, y, l=0.)
    return wc


def momentum(winit, X, y, steps=100, eta=0.01, tol=1e-6):
    vel_prev = wc = winit
    gamma = 0.9
    count = 0
    Gradient = gradient(wc, X, y, l=0.)
    while not gottol(Gradient, tol):
        count = count + 1
        vel = gamma*vel_prev + eta * gradient(wc, X, y)
        wc = wc - vel
        vel_prev = vel
        Gradient = gradient(wc, X, y, l=0.)
    return wc, count


def nesterov(winit, X, y, steps=100, eta=0.01, tol=1e-6):
    vel_prev = wc = winit
    gamma = 0.9
    count = 0
    Gradient = gradient(wc, X, y, l=0.)
    while not gottol(Gradient, tol):
        count = count + 1
        vel = gamma*vel_prev + eta * gradient(wc - gamma * vel_prev, X, y)
        wc = wc - vel
        print(wc)
        vel_prev = vel
        Gradient = gradient(wc, X, y, l=0.)
    return wc, count

# https://onlinecourses.science.psu.edu/stat462/node/101
df = pd.read_excel('test.xlsx')
X = np.array(df['X'])
X = np.array([[1]*len(X), X]).T
y = np.array(df['Y']).T

w = np.array([0,0])

normal_w = np.matmul(np.linalg.inv(np.matmul(X.T, X)), np.matmul(X.T, y))
print('Parameters from Normal Equation:', normal_w)

model = sm.OLS(y, X.astype(float)).fit()
print('Parameters from Stats Models', model.params)

w1 = goptimizer1(w, X, y, steps = 100000, eta=1e-3, tol=1e-6)
print('Parameters from vanilla (for loop):', w1)

w2 = goptimizer2(w, X, y, steps = 100000, eta=1e-3, tol=1e-6)
print('Parameters from vanilla (while loop):', w2)

w_v, c_v = goptimizer(w, X, y, steps = 1000, eta=1e-3, tol=1e-6)
print('Parameters from vanilla (sergey):', w_v)

w3, c3 = goptimizer3(w, X, y, steps = 100000, eta=1e-3, tol=1e-6)
print('Parameters from vanilla (modified sergey):', w3)

w_a, c_a = adoptimizer(w, X, y, steps = 100000,tol=1e-6)
print('Parameters from adadelta:', w_a)

w_b, c_b = btoptimizer(w, X, y, steps=1000000, eta=1e-3, tol=1e-6)
print('Parameters from backtracking:', w_b)

w_m, count = momentum(w, X, y, steps=0, eta=1e-2, tol=1e-6)
print('Parameters from momentum:', w_m, count)

w_n, count = nesterov(w, X, y, steps=0, eta=1e-2, tol=1e-6)
print('Parameters from nestorov:', w_n, count)
