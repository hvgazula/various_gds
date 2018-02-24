# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 10:29:51 2018

@author: Harsh
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import scipy as sp

# Pooled regression
df = pd.read_excel('test.xlsx')
X = np.array(df['X'])
X = np.array([[1]*len(X), X]).T
y = np.array(df['Y']).T

w = np.array([0,0])

normal_w = np.matmul(np.linalg.inv(np.matmul(X.T, X)), np.matmul(X.T, y))
print('Parameters from Normal Equation:', normal_w)

model = sm.OLS(y, X.astype(float)).fit()
print('Parameters from Stats Models', model.params)

df1 = pd.read_csv('site1.csv')
X1 = np.array(df1['X'])
X1 = np.array([[1]*len(X1), X1]).T
y1 = np.array(df1['Y']).T

df2 = pd.read_csv('site2.csv')
X2 = np.array(df2['X'])
X2 = np.array([[1]*len(X2), X2]).T
y2 = np.array(df2['Y']).T

df3 = pd.read_csv('site3.csv')
X3 = np.array(df3['X'])
X3 = np.array([[1]*len(X3), X3]).T
y3 = np.array(df3['Y']).T

df4 = pd.read_csv('site4.csv')
X4 = np.array(df4['X'])
X4 = np.array([[1]*len(X4), X4]).T
y4 = np.array(df4['Y']).T

# single-shot regression
model1 = sm.OLS(y1, X1.astype(float)).fit()
model2 = sm.OLS(y2, X2.astype(float)).fit()
model3 = sm.OLS(y3, X3.astype(float)).fit()
model4 = sm.OLS(y4, X4.astype(float)).fit()

sum_params = [model1.params, model2.params, model3.params, model4.params]
count_y_local = [len(y1), len(y2), len(y3), len(y4)]
avg_beta_vector = np.average(sum_params, weights=count_y_local, axis=0)
#avg_beta_vector = np.average(sum_params, axis=0)
print('Parameters from Single-shot: ', avg_beta_vector)

# multi-shot regression (normal equation)
cov1 = np.matmul(np.matrix.transpose(X1), X1)
cov2 = np.matmul(np.matrix.transpose(X2), X2)
cov3 = np.matmul(np.matrix.transpose(X3), X3)
cov4 = np.matmul(np.matrix.transpose(X4), X4)

Xy1 = np.matmul(np.matrix.transpose(X1), y1)
Xy2 = np.matmul(np.matrix.transpose(X2), y2)
Xy3 = np.matmul(np.matrix.transpose(X3), y3)
Xy4 = np.matmul(np.matrix.transpose(X4), y4)

avg_beta_vector = np.matmul(
    sp.linalg.inv(cov1 + cov2 + cov3 + cov4), Xy1) + np.matmul(
        sp.linalg.inv(cov1 + cov2 + cov3 + cov4), Xy2) + np.matmul(
            sp.linalg.inv(cov1 + cov2 + cov3 + cov4), Xy3) + np.matmul(
                sp.linalg.inv(cov1 + cov2 + cov3 + cov4), Xy4)

print('Parameters from multi-shot (normal eqn.): ', avg_beta_vector)

# multi-shot regression (gradient descent)
def gottol(vector, tol=1e-5):
    """Check if the gradient meets the tolerances"""
    return np.sum(np.square(vector))


def objective(weights, X, y, lamb=0.0):
    """calculates the Objective function value"""
    return (1 / 2 * len(X)) * np.sum(
        (np.dot(X, weights) - y)**2) + lamb * np.linalg.norm(
            weights, ord=2) / 2.


def gradient(weights, X, y, lamb=0.0):
    """Computes the gradient"""
    return (1 / len(X)) * np.dot(X.T, np.dot(X, weights) - y) + lamb * weights


##################### Vanilla Gradient Descent #####################
wp = np.zeros(X1.shape[1]) # Step 01
prev_obj_remote = np.inf # Step 01
tol = 1e-6 # Step 01
eta = 1e-3 # Step 01
grad_tol = 2 * tol # Step 01

count = 0
while grad_tol > tol: # Step 02
    count = count + 1

    # Step 05
    grad_local1 = gradient(wp, X1, y1, lamb=0)
    grad_local2 = gradient(wp, X2, y2, lamb=0)
    grad_local3 = gradient(wp, X3, y3, lamb=0)
    grad_local4 = gradient(wp, X4, y4, lamb=0)

    obj_local1 = objective(wp, X1, y1, lamb=0)
    obj_local2 = objective(wp, X2, y2, lamb=0)
    obj_local3 = objective(wp, X3, y3, lamb=0)
    obj_local4 = objective(wp, X4, y4, lamb=0)

    # at remote Step 08
    grad_remote = grad_local1 + grad_local2 + grad_local3 + grad_local4

    # Step 09
    wc = wp - eta * grad_remote

    # Step 10
    curr_obj_remote = obj_local1 + obj_local2 + obj_local3 + obj_local4

#    print('{:^15d} {:^20.6f} {:^20.6f} {:^15.5f} {:^15.7f}'.
#      format(count, prev_obj_remote, curr_obj_remote, eta,
#             np.sum(np.square(grad_remote))))

    # Step 11
    if curr_obj_remote > prev_obj_remote:

        # Step 12
        eta = eta/2
        print('halving')

    # Step 13
    else:

        # what is step 14???
#        grad_local1 = gradient(wc, X1, y1, lamb=0)
#        grad_local2 = gradient(wc, X2, y2, lamb=0)
#        grad_local3 = gradient(wc, X3, y3, lamb=0)
#        grad_local4 = gradient(wc, X4, y4, lamb=0)
#        grad_remote = grad_local1 + grad_local2 + grad_local3 + grad_local4

        # Step 15
        prev_obj_remote = curr_obj_remote

        # Step 16
        wp = wc

        grad_tol = gottol(grad_remote, tol)
        print(grad_tol)

#    if curr_obj_remote > prev_obj_remote:  # 11
#        eta = 0.5 * eta # 12
#        # start from scratch
#        wp = np.zeros(X1.shape[1])
#        prev_obj_remote = np.inf
#        grad_remote = np.random.rand(X1.shape[1])
#        if eta < 1e-6:
#            break
#        continue
#    else:  # 13
#        prev_prev = prev_obj_remote
#        prev_obj_remote = curr_obj_remote
#        wp = wc # 9
#
avg_beta_vector = wc

print('Parameters from multi-shot (gd): {}, iterations: {}'.format(avg_beta_vector, count))

######################## multi-shot regression (momentum) ####################
vel_prev = wc = np.zeros(X1.shape[1])
#prev_obj_remote = np.inf
grad_remote = np.random.rand(X1.shape[1])
tol = 1e-6
eta = 1e-2
gamma = 0.9

count = 0
while not gottol(grad_remote, tol):
    count = count + 1

    # At local
    grad_local1 = gradient(wc, X1, y1, lamb=0)
    grad_local2 = gradient(wc, X2, y2, lamb=0)
    grad_local3 = gradient(wc, X3, y3, lamb=0)
    grad_local4 = gradient(wc, X4, y4, lamb=0)

#    curr_obj_local1 = objective(wc, X1, y1, lamb=0)
#    curr_obj_local2 = objective(wc, X2, y2, lamb=0)
#    curr_obj_local3 = objective(wc, X3, y3, lamb=0)
#    curr_obj_local4 = objective(wc, X4, y4, lamb=0)
#
#    curr_obj_remote = curr_obj_local1 + curr_obj_local2 + curr_obj_local3 + curr_obj_local4

    # at remote
#    grad_remote = grad_local1 + grad_local2 + grad_local3 + grad_local4
    grad_local_vec = [grad_local1, grad_local2, grad_local3, grad_local4]
    count_y_local = [len(y1), len(y2), len(y3), len(y4)]
    grad_remote = np.average(grad_local_vec, weights=count_y_local, axis=0)

    vel = gamma*vel_prev + eta * grad_remote
    wc = wc - vel
    vel_prev = vel

#    if curr_obj_remote > prev_obj_remote:  # 11
##        eta = eta - eta * (25 / 100)  # 12
#        eta = 0.5 * eta # 12
#        # start from scratch
#        wp = np.zeros(X1.shape[1])
#        prev_obj_remote = np.inf
#        grad_remote = np.random.rand(X1.shape[1])
#        if eta < 1e-6:
#            break
#        continue
#    else:  # 13
#        prev_prev = prev_obj_remote
#        prev_obj_remote = curr_obj_remote
#        wp = wc # 9

avg_beta_vector = wc

print('Parameters from multi-shot (momentum): {}, iterations: {}'.format(avg_beta_vector, count))

