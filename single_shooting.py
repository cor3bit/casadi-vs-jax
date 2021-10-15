# Based on Casadi example
# https://github.com/casadi/casadi/blob/master/docs/examples/python/direct_single_shooting.py

import casadi as ca
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def run():
    # Model
    x1 = ca.MX.sym('x1')
    x2 = ca.MX.sym('x2')
    x3 = ca.MX.sym('x3')
    x4 = ca.MX.sym('x4')
    x = ca.vertcat(x1, x2, x3, x4)

    u = ca.MX.sym('u')

    xdot = ca.vertcat(
        x3,
        x4,
        (ca.sin(x2) * x4 * x4 + 9.81 * ca.sin(x2) * ca.cos(x2) + u) / (
                2 - ca.cos(x2) * ca.cos(x2)),
        -(ca.sin(x2) * ca.cos(x2) * x4 * x4 + 2 * 9.81 * ca.sin(x2) + ca.cos(x2) * u) / (
                2 - ca.cos(x2) * ca.cos(x2)),
    )

    # Objective term
    L = (x1 - 2) ** 2 + 0.01 * x2 ** 2 + 0.001 * x3 ** 2 + 0.001 * x4 ** 2 + 0.001 * u ** 2

    # Fixed step Runge-Kutta 4 integrator

    M = 10  # RK4 steps per interval
    T = 2.  # Time horizon
    N = 20  # number of control intervals
    DT = T / N / M

    f = ca.Function('f', [x, u], [xdot, L])

    X0 = ca.MX.sym('X0', 4)
    U = ca.MX.sym('U')
    X = X0
    Q = 0
    for j in range(M):
        k1, k1_q = f(X, U)
        k2, k2_q = f(X + DT / 2 * k1, U)
        k3, k3_q = f(X + DT / 2 * k2, U)
        k4, k4_q = f(X + DT * k3, U)
        X = X + DT / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        Q = Q + DT / 6 * (k1_q + 2 * k2_q + 2 * k3_q + k4_q)

    F = ca.Function('F', [X0, U], [X, Q], ['x0', 'p'], ['xf', 'qf'])

    # Start with an empty NLP
    w = []
    w0 = []
    lbw = []
    ubw = []
    J = 0

    # Formulate the NLP
    Xk = ca.MX([0, 0, 0, 0])
    for k in range(N):
        # New NLP variable for the control
        Uk = ca.MX.sym('U_' + str(k))
        w += [Uk]
        lbw += [-20]
        ubw += [20]
        w0 += [0]

        # Integrate till the end of the interval
        Fk = F(x0=Xk, p=Uk)
        Xk = Fk['xf']
        J = J + Fk['qf']

    # Add inequality constraint
    Xf = ca.MX([2, 0, 0, 0])
    g = []
    lbg = []
    ubg = []
    g += [Xk - Xf]
    lbg += [0, 0, 0, 0]
    ubg += [0, 0, 0, 0]

    # Create an NLP solver
    prob = {
        'f': J,
        'x': ca.vertcat(*w),
        'g': ca.vertcat(*g)
    }
    solver = ca.nlpsol('solver', 'ipopt', prob)

    # Solve the NLP
    sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
    w_opt = sol['x']

    print(w_opt)

    # plotting
    fig, ax = plt.subplots()

    w_k = np.array(w_opt)
    w_k_dims = w_k.shape[0]

    times = np.linspace(0, 2, w_k_dims + 1)
    w_k_steps = np.concatenate((w_k, [w_k[-1]]))

    ax.plot(times, np.array([-20 for i in range(len(times))]), color='red', ls='--')
    ax.plot(times, np.array([20 for i in range(len(times))]), color='red', ls='--')
    ax.step(times, w_k_steps, where='post')

    plt.xlabel('$t$')
    plt.ylabel('$u$')
    plt.show()


# -------------------------- Runner --------------------------

if __name__ == '__main__':
    run()
