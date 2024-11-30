import math
import numpy as np
import matplotlib.pyplot as plt

class Iteration:
    def __init__(self, g):
        self.g = g
    def run(self, x0, k):
        l = [x0]
        for i in range(k):
            x0 = self.g(x0)
            l.append(x0)
        return l
    def plot(self, x0, k, interval):
        fig, ax = plt.subplots(figsize=(5, 5), layout='constrained')

        x = np.linspace(*interval, 100)
        y = self.g(x)
        ax.plot(x, y, label='$g$')
        ax.plot(x, x, label='$y=x$')

        xk = np.array(self.run(x0, k))
        k = len(xk)
        ax.scatter(xk[1:-1], self.g(xk)[1:-1], zorder=3)
        ax.scatter(xk[0], self.g(xk[0]), c='r', label='start', zorder=3)
        ax.scatter(xk[-1], self.g(xk[-1]), c='y', label='end', zorder=3)

        x = []
        y = []
        for xi in xk:
            x.append(xi)
            y.append(self.g(xi))
            x.append(self.g(xi))
            y.append(self.g(xi))
        ax.plot(x[:-1], y[:-1], ls='--', c='k', label='cobwebbing')
        
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_aspect('equal')
        ax.grid('on')
        ax.legend()
        ax.set_xlim(interval)
        ax.set_ylim(interval)
        
        plt.show()
    def plot_speed(self, x0, k, realAnswer):
        y = np.array(self.run(x0, k))
        k = len(y)
        
        x = np.arange(k)
        y = -np.log10(abs(realAnswer - y))
        
        fig, ax = plt.subplots(1, 2, figsize=(10, 5), layout='constrained')
        ax[0].plot(x, y)
        ax[0].set_xlabel('Number of iterations')
        ax[0].set_ylabel('Digits of precision')
        ax[0].grid('on')
        
        ax[1].plot(x[:-1], y[1:] - y[:-1])
        ax[1].set_xlabel('Number of iterations')
        ax[1].set_ylabel('Digits of precision gained')
        ax[1].grid('on')
        plt.show()

class Newton(Iteration):
    def __init__(self, f, df, max_val=1e3):
        self.f = f
        self.df = df
        self.g = lambda x: x - f(x) / df(x)
    def nplot(self, x0, k, intervalx, intervaly):
        fig, ax = plt.subplots(figsize=(5, 5), layout='constrained')

        x = np.linspace(*intervalx, 1000)
        y = self.f(x)
        ax.plot(x, y, lw=2, label='$f$')
        ax.axhline(0, lw=2, c='k', zorder=1)
        ax.axvline(0, lw=2, c='k', zorder=1)

        xk = np.array(self.run(x0, k))
        k = len(xk)
        zero = np.zeros(k - 2)
        ax.scatter(xk[1:-1], zero, zorder=4)
        ax.scatter(xk[0], 0, c='r', label='start', zorder=4)
        ax.scatter(xk[-1], 0, c='y', label='end', zorder=4)
        ax.scatter(xk, self.f(xk), zorder=3)

        for xi in xk[:-1]:
            ax.plot(x, self.df(xi) * (x - xi) + self.f(xi), ls=':', c='r', zorder=1)
        for xi in xk:
            ax.plot([xi, xi], [0, self.f(xi)], ls='--', c='k', zorder=0)
        
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_aspect('equal')
        ax.grid('on')
        ax.legend()

        ax.set_xlim(intervalx)
        ax.set_ylim(intervaly)
        
        plt.show()

def sign(x):
    if x < 0:
        return -1
    if x > 0:
        return 1
    return 0

class Bisection:
    def __init__(self, f):
        self.f = f
    def run(self, I, k):
        l = []
        
        a, b = I
        fa, fb = self.f(a), self.f(b)
        if fa == 0:
            return [a]
        elif fb == 0:
            return [b]
        elif sign(fa) == sign(fb):
            return None
        
        for i in range(k + 1):
            x = (a + b) / 2
            f = self.f(x)
            if f == 0:
                return l

            if sign(f) != sign(fa):
                b = x
                fb = f
            else:
                a = x
                fa = f
            l.append(x)
        return l
    def plot(self, I, k, intervaly):
        fig, ax = plt.subplots(figsize=(5, 5), layout='constrained')

        x = np.linspace(*I, 1000)
        y = self.f(x)
        ax.plot(x, y, lw=2, label='$f$')
        ax.axhline(0, lw=2, c='k', zorder=1)
        ax.axvline(0, lw=2, c='k', zorder=1)

        xk = np.array(self.run(I, k))
        k = len(xk)
        zero = np.zeros(k)
        ax.scatter(xk, zero, c=np.arange(k), s=20, zorder=4)
        ax.scatter(I[0], 0, c='r', label='$a_0$', zorder=4)
        ax.scatter(I[1], 0, c='y', label='$b_0$', zorder=4)
        ax.scatter(xk, self.f(xk), c=np.arange(k), s=20, zorder=3)

        for x in xk:
            ax.plot([x, x], [0, self.f(x)], ls=':', c='r', zorder=1)
        
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_aspect('equal')
        ax.grid('on')
        ax.legend()

        ax.set_xlim(I)
        ax.set_ylim(intervaly)
        
        plt.show()

def vec(x, y):
    return np.array([[x], [y]])
def unvec(v):
    return v[0, 0], v[1, 0]
def norm(v):
    x, y = unvec(v)
    return max(abs(x), abs(y))

class Newton2D(Iteration):
    def __init__(self, *args):
        if len(args) == 3:
            f1 = args[0]
            f2 = args[1]
            J = args[2]

            self.f = lambda v: vec(f1(v[0, 0], v[1, 0]), f2(v[0, 0], v[1, 0]))
            self.J = lambda v: J(v[0, 0], v[1, 0])
            self.g = lambda v: v - np.matmul(np.linalg.inv(self.J(v)), self.f(v))
        else:
            f = args[0]
            J = args[1]

            self.f = f
            self.J = J
            self.g = lambda v: v - np.matmul(np.linalg.inv(self.J(v)), self.f(v))
    def plot(self, v0, k, intervalx, intervaly, ans=-1):
        fig, ax = plt.subplots(figsize=(5, 5), layout='constrained')
        
        v = self.run(v0, k)
        k = len(v)
        x = [vk[0, 0] for vk in v]
        y = [vk[1, 0] for vk in v]

        ax.scatter(x, y, s=10, c=range(k))
        if not isinstance(ans, int):
            ax.axhline(ans[1, 0], ls='--', lw=1, c='k')
            ax.axvline(ans[0, 0], ls='--', lw=1, c='k')
            # ax.scatter(ans[0, 0], ans[1, 0], s=10, c='b')
        
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_aspect('equal')
        ax.grid('on')

        ax.set_xlim(intervalx)
        ax.set_ylim(intervaly)
        
        plt.show()
        




