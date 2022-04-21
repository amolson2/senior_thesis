from pybeebase.integrator import RKF54
from pybeeastro.bodies import Earth, Moon
# from pybeeastro.eom.pcr3bp import PCR3BP
from pybeeastro.eom.cr3bp import CR3BP
from pybeeastro.eom.twobody import TwoBody
from pybeeastro.astrodynamics import coe2rv, rv2coe, convert_state_from_primary_centered_2BP_to_CR3BP
from numpy import array, concatenate, ndarray, pi, sqrt, load, argsort, zeros, empty, reshape, arange, savez
from numpy.linalg import norm
from scipy.optimize import Bounds, NonlinearConstraint, BFGS, minimize
import matplotlib.pyplot as plt

num_dimensions = 3
solution_history = []
earth = Earth()
moon = Moon()
initial_alt = 200000.
final_alt = 300000.

constraint_tol = 0.0001
max_dv = 1.


def get_orbital_period(semimajor: float,
                       mu: float) -> float:
    return 2 * pi * sqrt(((semimajor ** 3) / mu))


eom = CR3BP(primary=earth, secondary=moon)
initial_period = get_orbital_period(earth.equatorial_radius + initial_alt, earth.mu_for_children) / eom.normalized_units['TimeUnit']
hohmann_semimajor = ((2 * earth.equatorial_radius) + initial_alt + final_alt) / 2
hohmann_period = (get_orbital_period(hohmann_semimajor, earth.mu_for_children) / 2) / eom.normalized_units['TimeUnit']

npzfile = load('threebody_SLSQP.npz')
npzfile1 = load('threebody_SLSQP_2.npz')

solutions = concatenate((npzfile['arr_0'], npzfile1['arr_0']), axis=0)
coords = concatenate((npzfile['arr_1'], npzfile1['arr_1']), axis=0)
feasibility = concatenate((npzfile['arr_2'], npzfile1['arr_2']), axis=0)
optimality = concatenate((npzfile['arr_3'], npzfile1['arr_3']), axis=0)

feasible_coords = coords[feasibility]
feasible_solutions = solutions[feasibility]
optimal_coords = coords[optimality]
optimal_solutions = solutions[optimality]


final_solution_history = empty((len(feasible_solutions), 6))
final_optimality = empty(len(feasible_solutions), dtype=bool)
final_feasibility = empty(len(feasible_solutions), dtype=bool)


def get_initial_state(alt):
    classical_orbital_elements = array([earth.equatorial_radius + alt, 0., 0., 0., 0., 0.])
    position, velocity = coe2rv(classical_orbital_elements=classical_orbital_elements, mu=earth.mu_for_children)
    semimajor = classical_orbital_elements[0]
    period = 2 * pi * sqrt(((semimajor ** 3) / earth.mu_for_children))
    return position, velocity, period


def twobody_to_threebody(position, velocity, period):
    state = concatenate((position, velocity))
    eom = CR3BP(primary=earth, secondary=moon)
    period /= eom.normalized_units['TimeUnit']
    transformed_state = convert_state_from_primary_centered_2BP_to_CR3BP(state, earth, moon)
    return transformed_state[0:3], transformed_state[3:6], period


def objective_function(control) -> float:
    # print('cost: {}'.format(norm(control)))
    cost = norm(control[0:4])
    # solution_history.append(control)
    return cost


for i in range(len(feasible_solutions)):
    initial_control = concatenate((optimal_solutions[i], array(optimal_coords[i])))

    def simulation_3b(control: ndarray) -> ndarray:
            eom = CR3BP(primary=earth, secondary=moon)
            # get initial state and convert to three body frame
            position, velocity, period = get_initial_state(initial_alt)
            position, velocity, period = twobody_to_threebody(position, velocity, period)
            state = concatenate((position, velocity))

            # propagate for initial time
            rk54 = RKF54()
            rk54.add_drift_vector_field(vector_field=eom.evaluate)
            rk54.evaluate(s=state, t0=0., tf=control[-2])

            # add initial burn
            position = rk54.states[-1][0:3]
            velocity = rk54.states[-1][3:6]
            velocity[0:2] += control[0:2]
            state = concatenate((position, velocity))

            # propagate transfer ellipse
            rk54 = RKF54()
            rk54.add_drift_vector_field(vector_field=eom.evaluate)
            rk54.evaluate(s=state, t0=0., tf=control[-1])

            # add the final burn to the last state
            rk54.states[-1][3:5] += control[2:4]
            return rk54.states


    def final_constraint_3b(control) -> float:
            final_state = simulation_3b(control)[-1]

            # determine final orbit in three body coordinates
            position, velocity, period = get_initial_state(final_alt)
            position, velocity, period = twobody_to_threebody(position, velocity, period)

            # propagate for half orbit
            state = concatenate((position, velocity))
            eom = CR3BP(primary=earth, secondary=moon)
            rk54 = RKF54()
            rk54.add_drift_vector_field(vector_field=eom.evaluate)
            rk54.evaluate(s=state, t0=0., tf=control[-2]+control[-1])
            desired_state = rk54.states[-1]
            error = norm((final_state - desired_state) / norm(desired_state))
            # print('three body error: {}'.format(error))
            return error


    lower_bounds = [-max_dv, -max_dv, -max_dv, -max_dv, 0, 0]
    upper_bounds = [max_dv, max_dv, max_dv, max_dv, initial_period*5, hohmann_period*6]
    bounds = Bounds(lb=lower_bounds, ub=upper_bounds)

    nonlinear_constraint_3b = NonlinearConstraint(final_constraint_3b, -constraint_tol, constraint_tol, jac='2-point', hess=BFGS())

    result_3b = minimize(objective_function, initial_control, jac='2-point', hess=BFGS(),
                        constraints=[nonlinear_constraint_3b],
                        options={'disp': 1, 'maxiter': 500},
                        bounds=bounds)
    print(result_3b.x)
    print(norm(result_3b.x[0:4]))

    final_solution_history[i] = result_3b.x
    if result_3b.success:
        final_feasibility[i] = True
        final_optimality[i] = True
    else:
        if final_constraint_3b(result_3b.x) < constraint_tol:
            final_feasibility[i] = True
            final_optimality[i] = False
        else:
            final_feasibility[i] = False
            final_optimality[i] = False

savez('verification_results_300000_1e-4', final_solution_history, feasible_coords, final_feasibility, final_optimality)
