import numpy as np
import ipdb

def objective(w, X, y, l=0.0):
    return np.sum((y - np.dot(X,w))**2) + l * np.dot(w.T,w)/2.

def gradient(w, X, y, l=0.0):
    return -2.0 * np.dot(X.T, y - np.dot(X,w)) + l * w


def gottol(v, tol=1e-5):
    return np.sum(np.square(v)) <= tol

eta = 0.1

X = np.array([[1, 2, 3, 4, 5],[1,1,1,1,1]]).T
y = np.array([11, 12, 13, 14, 15]).T


X    =    [    -4.67339189e+02,    -1.91739189e+02,    2.93060811e+02,
               -2.20391892e+01,  -1.55391892e+01,  -6.81391892e+01,  -1.60039189e+02,
               7.18860811e+02,   1.38760811e+02,    9.15608108e+01,   3.39608108e+01,
               2.36608108e+01,   -1.10293919e+03,   2.43160811e+02,   5.34608108e+01,
               2.99960811e+02,   9.74608108e+01,   -1.01313919e+03,   3.07660811e+02,
               -2.61391892e+01,   1.35296081e+03,  -3.95639189e+02,   4.15860811e+02,
               -6.89391892e+01,  -6.79739189e+02,   -4.61839189e+02,  3.50560811e+02,
               -5.87639189e+02,  -4.07339189e+02,   -8.85639189e+02,  3.05760811e+02,
               5.07560811e+02,   -4.52439189e+02,  -3.24391892e+01,   6.90760811e+02,
               -8.69391892e+01, 9.60810811e-01,   389.56081081,   299.96081081,
               267.36081081,     6.26081081, -1013.13918919,  -233.03918919,   -86.93918919,
               18.96081081, -467.33918919,   305.76081081,   690.76081081,    61.46081081,
               -452.43918919,   385.56081081,   145.86081081,    97.46081081,
               -26.13918919,   507.56081081,   -68.13918919,   307.66081081,
               -32.43918919,   -68.93918919,  -983.73918919,    23.66081081,
               53.46081081,   815.06081081,   293.06081081, -1683.23918919,
               -419.53918919,  -191.73918919,  -587.63918919,  -114.03918919,
               350.56081081, -1102.93918919,   138.76081081,  -525.83918919,
               -673.33918919 ]

X = np.array([X,[1]*len(X)]).T
y = [ 1,  1, -1, -1, -1, -1,  1, -1,  1,  1, -1, -1,  1, -1, -1,  1,  1,
      -1, -1, -1, -1,  1, -1, -1,  1, -1,  1, -1, -1,  1, -1, -1,  1,  1,
      -1,  1,  1, -1,  1, -1, -1,  1, -1,  1,  1, -1, -1, -1,  1,  1, -1,  1,  1,  1,
      -1, -1, -1,  1, -1,  1,  1,  1,  1, -1,  1, -1,  1,  1, -1, -1,  1,
      -1, -1,  1]
y = np.array(y).T


w = np.array([1,1])



backtrack = False

eta = 1e-1
rho = 0.99
eps = 1e-5
tol = 1e-6

wc = wp = w
Eg2 = 0
EdW2 = 0
co = []

Gradient = gradient(wc, X, y)
pObj = 1e10
#pObj = objective(wc,X,y)

for i in range(40):
    cObj = objective(wc,X,y)

    if cObj > pObj:
        eta /= 2.
        ipdb.set_trace()
        backtrack = True
        print 'adjusting: ', eta, cObj #wc, wp,  Gradient

    if backtrack:
        wc = wp - eta * Gradient
        backtrack = False
    else:
        Gradient = gradient(wc, X, y, l=0.)
        wc = wp - eta * Gradient
        wp = wc
        pObj = cObj
    print eta, cObj

stop


for i in range(8000):
    cObj = objective(wc,X,y,l=0.)
    if cObj > pObj: break

    Gradient = gradient(wc, X, y,l=0.)
    if gottol(Gradient, tol): break

    # gradient descent
    #wc = wc - eta * Gradient

    # ADADELTA
    Eg2 = rho*Eg2 + (1 - rho) * Gradient * Gradient
    dW = - np.sqrt(EdW2+eps)/np.sqrt(Eg2+eps) *Gradient
    EdW2 = rho*EdW2 + (1 - rho) * dW * dW
    wc = wc + dW

    co.append(cObj)
    pObj = cObj
    wp = wc
    print i, cObj, Gradient
