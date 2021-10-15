import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

VERBOSE = 0


def problem4():
    x = ca.SX.sym('x')
    y = ca.SX.sym('y')

    nlp_params = {
        'x': ca.vertcat(x, y),
        'f': 0.5 * x * x + 0.5 * y * y + x + y,
        'g': x * x + y * y - 1,
    }

    S = ca.nlpsol('S', 'ipopt', nlp_params, {'verbose': False, 'ipopt.print_level': VERBOSE, 'print_time': VERBOSE})

    # solve
    pt_names = ['A1', 'A2', 'A3', 'B1', 'B2', 'C', 'D', ]

    initial_guesses = [
        (0., 1.), (-1., -1.), (-1., 1.), (1., 1.), (1., 1. + 1e-6), (0, 0), (0.5, 1.),
    ]

    res = {}

    for pt_name, guess in zip(pt_names, initial_guesses):
        r = S(x0=guess, lbg=0, ubg=0)
        x_opt = r['x']
        sol = r['f']
        # print('x_opt: ', x_opt)
        res[pt_name] = (x_opt, sol)

    print(res)


def problem5():
    x = ca.SX.sym('x')
    y = ca.SX.sym('y')
    z = ca.SX.sym('z')

    constr1 = x * x - 2 * y * y * y - y - 10 * z
    constr2 = y + 10 * z

    g = ca.vertcat(constr1, constr2)

    nlp_params = {
        'x': ca.vertcat(x, y, z),
        'f': 0.5 * x * x + 0.5 * y * y + 0.5 * z * z + y,
        'g': g,
    }

    S = ca.nlpsol('S', 'ipopt', nlp_params, {'verbose': False, 'ipopt.print_level': VERBOSE, 'print_time': VERBOSE})

    # solve
    guess = (1, 1, 0)

    res = {}

    r = S(x0=guess, lbg=0, ubg=0)
    x_opt = r['x']
    sol = r['f']
    # print('x_opt: ', x_opt)
    res['A'] = (x_opt, sol)

    print(res)



def run():
    problem4()
    problem5()



# -------------------------- Runner --------------------------

if __name__ == '__main__':
    run()
