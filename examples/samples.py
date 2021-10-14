from casadi import *

# Create scalar/matrix symbols
x = MX.sym('x', 5)

# Compose into expressions
y = norm_2(x)

# Sensitivity of expression -> new expression
grad_y = gradient(y, x)

# Create a Function to evaluate expression
f = Function('f', [x], [grad_y])

# Evaluate numerically
grad_y_num = f([1, 2, 3, 4, 5])


print(grad_y_num)