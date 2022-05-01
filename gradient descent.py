from linear_algebra import Vector, dot
from typing import Callable, List

def square(x):
    return x*x

def derivative(x):
    return 2*x

def sum_of_squares(v: List)->float:
    return sum(x_1*y_1 for x_1,y_1 in zip(v,v))

def difference_quotient(f:Callable[[float], float], x:float, h:float)->float:
    return (f(x+h)-f(x))/h

xs = range(-10,11)
actuals = [derivative(x) for x in xs]
estimates = [difference_quotient(square, x, h=0.001) for x in xs]

import matplotlib.pyplot as plt
plt.plot(xs,actuals,'rx')
plt.plot(xs, estimates, 'b+')
plt.show()


def partial_difference_quotient(f: Callable[[List], float], v, i, h):
    w = [v_j + (h if i==j else 0) for j, v_j in enumerate(v)]
    return (f(w) - f(v)) / h

def estimate_gradient(f: Callable[[List], float], v:List, h):
    # for every element of the list we find its partial derivative
    return [partial_difference_quotient(f,v,i,h) for i in range(len(v))]

import random

def gradient_step(v: List, gradient: List, step_size: float):
    """Moves 'step_size' in the 'gradient' direction from 'v'"""
    assert len(v) == len(gradient)
    # multiply vector with scalar
    step = [step_size * v_i for v_i in gradient]
    # sum of vectors
    return [v_0 + s_0 for v_0, s_0 in zip(v, step)]

def sum_of_squares_gradient(v):
    return [2*v_0 for v_0 in v]

#random start
v = [random.uniform(-10, 10) for i in range(3)]
print(v)

for epoch in range(1000):
    grad = sum_of_squares_gradient(v)   # compute gradient at v
    v = gradient_step(v ,grad, -0.01)
    print(epoch, v)

inputs = [(x, 20 * x + 5) for x in range(-50, 50)]

def vector_sum(vectors):
    num_elements = len(vectors[0])
    return [sum(vector[i] for vector in vectors)
            for i in range(num_elements)]

def scalar_multiply(c, v):
    """Multiplies every element by c"""
    return [c * v_i for v_i in v]

def vector_mean(vectors: List[Vector]) -> Vector:
    """Computes the element-wise average"""
    n = len(vectors)
    return scalar_multiply(1/n, vector_sum(vectors))

def linear_gradient(x, y, theta):
    slope, intercept = theta
    predicted = slope * x + intercept
    error = predicted - y
    squared_error = error**2
    grad = [2 * error * x, 2 * error]
    return grad

theta = [random.uniform(-1,1), random.uniform(-1,1)]
learning_rate = 0.001

for epoch in range(5000):
    # Compute the mean of the gradients
    grad = vector_mean([linear_gradient(x,y,theta) for x,y in inputs])
    # Take a step in that direction
    theta = gradient_step(theta, grad, -learning_rate)
    print(epoch, theta)

slope, intercept = theta
print(str(slope) + " " + str(intercept))

from typing import TypeVar, List, Iterator

T = TypeVar('T')

def minibatches(dataset: List[T], batch_size: int, shuffle: bool=True):
    batch_starts = [start for start in range(0, len(dataset), batch_size)]
    if shuffle: random.shuffle(batch_starts)

    for start in batch_starts:
        end = start + batch_size
        yield dataset[start:end]


theta = [random.uniform(-1,1), random.uniform(-1, 1)]

for epoch in range(1000):
    for batch in minibatches(inputs, batch_size=20):
        grad = vector_mean([linear_gradient(x, y, theta) for x, y in batch])
        theta = gradient_step(theta, grad, -learning_rate)
    print(epoch, theta)

slope, intercept = theta
print(slope, ' ', intercept)
