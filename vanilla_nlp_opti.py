import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

VERBOSE = 0


def problem4():
    opti = ca.Opti()

    x = opti.variable()
    y = opti.variable()

    opti.minimize(0.5 * x * x + 0.5 * y * y + x + y)

    opti.subject_to(x * x + y * y == 1)

    opts = {'ipopt.print_level': VERBOSE, 'print_time': VERBOSE}
    opti.solver('ipopt', opts)

    # solve
    pt_names = ['A1', 'A2', 'A3', 'B1', 'B2', 'C', 'D', ]

    initial_guesses = [
        (0., 1.), (-1., -1.), (-1., 1.), (1., 1.), (1., 1. + 1e-6), (0, 0), (0.5, 1.),
    ]

    res = {}

    for pt_name, guess in zip(pt_names, initial_guesses):
        init_x, init_y = guess
        opti.set_initial(x, init_x)
        opti.set_initial(y, init_y)
        sol = opti.solve()
        print(sol.stats()['iter_count'])
        res[pt_name] = (sol.value(x), sol.value(y))

    print(res)


def problem5():
    opti = ca.Opti()

    x = opti.variable()
    y = opti.variable()
    z = opti.variable()

    opti.minimize(0.5 * x * x + 0.5 * y * y + 0.5 * z * z + y)

    opti.subject_to(x * x - 2 * y * y * y - y - 10 * z == 0)
    opti.subject_to(y + 10 * z == 0)

    opts = {'ipopt.print_level': VERBOSE, 'print_time': VERBOSE}
    opti.solver('ipopt', opts)

    opti.set_initial(x, 1)
    opti.set_initial(y, 1)
    opti.set_initial(z, 0)
    sol = opti.solve()

    print((sol.value(x), sol.value(y), sol.value(z)))
    print(sol.stats()['iter_count'])


def problem7():
    opti = ca.Opti()

    x = opti.variable()
    y = opti.variable()

    opti.minimize(0.5 * x * x + 0.5 * y * y + x + y)

    opti.subject_to(x * x + y * y == 1)
    opti.subject_to(0.5 - x * x - y <= 0)

    opts = {'ipopt.print_level': VERBOSE, 'print_time': VERBOSE}
    opti.solver('ipopt', opts)

    # solve
    pt_names = ['A', 'B', 'C', 'D', 'E']

    initial_guesses = [
        (0., 1.), (-1., -1.), (0.9, 1.), (1., -1.), (0, -1),
    ]

    res = {}

    for pt_name, guess in zip(pt_names, initial_guesses):
        init_x, init_y = guess
        opti.set_initial(x, init_x)
        opti.set_initial(y, init_y)
        sol = opti.solve()
        print(sol.stats()['iter_count'])

        res[pt_name] = (sol.value(x), sol.value(y))

    print(res)


def run():
    problem4()
    problem5()
    problem7()


# -------------------------- Runner --------------------------

if __name__ == '__main__':
    run()
