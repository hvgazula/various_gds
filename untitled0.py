# -*- coding: utf-8 -*-
"""
Created on Sat Jan  6 19:17:51 2018

@author: Harsh
"""

import numpy as np
import pandas as pd

def objective(w, X, y, l=0.0):
    return np.sum((y - np.dot(X,w))**2) + l * np.dot(w.T,w)/2.


def gradient(w, X, y, l=0.0):
    return -2.0 * np.dot(X.T, y - np.dot(X,w)) + l * w


def gottol(v, tol=1e-1):
    return np.sum(np.square(v)) <= tol


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


def goptimizer1(winit, X, y, steps=1000, eta=0.01, tol=1e-3):
    wc = winit

    for i in range(steps):
        Gradient = gradient(wc, X, y, l=0.)
        wc = wc - eta * Gradient
    return wc


def goptimizer2(winit, X, y, steps=100, eta=0.01, tol=1e-3):
    wc = winit

    Gradient = gradient(wc, X, y, l=0.)
    while not gottol(Gradient, tol):
        wc = wc - eta * Gradient
        Gradient = gradient(wc, X, y, l=0.)
    return wc


def goptimizer(winit, X, y, steps=50000, eta=0.01, tol=1e-3):
    wc = wp = winit
    pObj = objective(wc, X, y)
    co = []

    for i in range(steps):
#    Gradient = gradient(wc, X, y)
#    while not gottol(Gradient, tol):
        cObj = objective(wc, X, y, l=0.)
        if cObj > pObj:
            eta = eta/2
            continue

        Gradient = gradient(wc, X, y, l=0.)
        if gottol(Gradient, tol):
            break
        wc = wc - eta * Gradient
        co.append(cObj)

        pObj = cObj
        wp = wc
    return wc, co


#def adoptimizer(winit, X, y, grad=gradient, obj=objective, steps=500,
#                rho=0.99, eps=1e-4, tol=1e-5):
#    wc = wp = winit
#    pObj = obj(wc, X, y)
#    Eg2 = 0
#    EdW2 = 0
#    co = []
#
#    for i in range(steps):
#        cObj = objective(wc, X, y, l=0.)
#        if cObj > pObj: break
#
#        Gradient = gradient(wc, X, y, l=0.)
##        print(Gradient)
#        if gottol(Gradient, tol): break
#
#        # ADADELTA
#        Eg2 = rho * Eg2 + (1 - rho) * Gradient * Gradient
#        dW = - np.sqrt(EdW2 + eps) / np.sqrt(Eg2 + eps) * Gradient
#        EdW2 = rho * EdW2 + (1 - rho) * dW * dW
#        wc = wc + dW
#        co.append(cObj)
#
#        pObj = cObj
#        wp = wc
#
#    return wc, co

# https://onlinecourses.science.psu.edu/stat462/node/101
df = pd.read_excel('test.xlsx')
X = np.array(df['X'])
X = np.array([[1]*len(X), X]).T
y = np.array(df['Y']).T

w = np.array([1,1])

#w_g = goptimizer1(w, X, y, eta=1e-3)
#print(w_g)

#w_g = goptimizer2(w, X, y, eta=1e-3)
#print(w_g)

w_g, co_g= goptimizer(w, X, y, eta=0.0001)
print(w_g)
#print(co_g)

#w_b, co_b = btoptimizer(w, X, y, eta=1e-4)
#print(w_b)

#w_a, co_a = adoptimizer(w, X, y)
#print(w_a)
#print(co_a)

#import statsmodels.api as sm
#model = sm.OLS(y, X.astype(float)).fit()
#print(model.params)

#print(np.matmul(np.linalg.inv(np.matmul(X.T, X)), np.matmul(X.T, y)))