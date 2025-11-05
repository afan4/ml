# pip install scikit-learn pandas numpy matplotlib

import numpy as np
import matplotlib.pyplot as plt

# Take user input for coefficients of quadratic ax^2 + bx + c
a = float(input("Enter coefficient a: "))
b = float(input("Enter coefficient b: "))
c = float(input("Enter coefficient c: "))
x0 = float(input("Enter starting x: "))
lr = float(input("Enter learning rate: "))
iters = int(input("Enter number of iterations: "))

# Define function and gradient
def f(x):
    return a * x**2 + b * x + c

def grad(x):
    return 2 * a * x + b

# Gradient Descent
x = x0
path = [x]

for _ in range(iters):
    x = x - lr * grad(x)
    path.append(x)

# Plot
x_plot = np.linspace(x0 - 10, x0 + 10, 100)
y_plot = f(x_plot)

plt.plot(x_plot, y_plot, label='f(x)')
plt.plot(path, f(np.array(path)), 'ro-', label='GD path')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Gradient Descent on Quadratic')
plt.legend()
plt.grid(True)
plt.show()

print(f"\nLocal minimum at x ≈ {path[-1]}, y ≈ {f(path[-1])}")

# OUTPUT
# Enter coefficient a: 1
# Enter coefficient b: 5
# Enter coefficient c: 5
# Enter starting x: 10
# Enter learning rate: 0.9
# Enter number of iterations: 10
# Local minimum at x ≈ -1.1578227199999986, y ≈ 0.5514398509482019



# Engineered By Afan Shaikh Git: github.com/afan4