import casadi as ca
import numpy as np
import matplotlib.pyplot as plt


def problem2():
    x = ca.SX.sym('x')
    y = ca.SX.sym('y')

    nlp_params = {
        'x': ca.vertcat(x, y),
        'f': 0.5 * x * x + 0.5 * y * y + x + y,
        'g': x * x + y * y - 1,
    }

    S = ca.nlpsol('S', 'ipopt', nlp_params)

    # solve
    pt_names = ['A1', 'A2', 'A3', 'B1', 'B2', 'C', 'D', ]

    initial_guesses = [
        (0., 1.), (-1., -1.), (-1., 1.), (1., 1.), (1., 1. + 1e-6), (0, 0), (0.5, 1.),
    ]

    res = {}

    for pt_name, guess in zip(pt_names, initial_guesses):
        r = S(x0=guess, lbg=0, ubg=0)
        x_opt = r['x']
        # print('x_opt: ', x_opt)
        res[pt_name] = x_opt

    print(res)


def run():
    problem2()


# -------------------------- Runner --------------------------

if __name__ == '__main__':
    run()
