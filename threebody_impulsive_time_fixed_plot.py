from pybeebase.integrator import RKF54
from pybeeastro.bodies import Earth, Moon
from pybeeastro.eom.cr3bp import CR3BP
from pybeeastro.astrodynamics import coe2rv, convert_state_from_primary_centered_2BP_to_CR3BP
from numpy import array, concatenate, ndarray, pi, sqrt, linspace, empty, ones, savez
from numpy.linalg import norm
from scipy.optimize import Bounds, NonlinearConstraint, BFGS, minimize

num_dimensions = 3
earth = Earth()
moon = Moon()
initial_alt = 200000.
final_alt = 300000.
hohmann_semimajor = ((2*earth.equatorial_radius)+initial_alt+final_alt)/2
constraint_tol = 0.001
max_dv = 3.


def get_orbital_period(semimajor: float,
                       mu: float) -> float:
    return 2 * pi * sqrt(((semimajor ** 3) / mu))


def get_initial_state(alt):
    classical_orbital_elements = array([earth.equatorial_radius + alt, 0., 0., 0., 0., 0.])
    position, velocity = coe2rv(classical_orbital_elements=classical_orbital_elements, mu=earth.mu_for_children)
    semimajor = classical_orbital_elements[0]
    period = get_orbital_period(semimajor, earth.mu_for_children)
    return position, velocity, period


def twobody_to_threebody(position, velocity, period):
    state = concatenate((position, velocity))
    eom = CR3BP(primary=earth, secondary=moon)
    period /= eom.normalized_units['TimeUnit']
    transformed_state = convert_state_from_primary_centered_2BP_to_CR3BP(state, earth, moon)
    return transformed_state[0:3], transformed_state[3:6], period


initial_period_2b = get_orbital_period(earth.equatorial_radius+initial_alt, earth.mu_for_children)
hohmann_period_2b = get_orbital_period(hohmann_semimajor, earth.mu_for_children)/2
eom = CR3BP(primary=earth, secondary=moon)
initial_period_3b = initial_period_2b / eom.normalized_units['TimeUnit']
hohmann_period_3b = hohmann_period_2b / eom.normalized_units['TimeUnit']
initial_time = linspace(0, 3*initial_period_3b, num=25).astype(float)
transfer_time = linspace(0.25*hohmann_period_3b, 4*hohmann_period_3b, num=25).astype(float)
final_solution_history = empty((len(initial_time)*len(transfer_time), 4))
feasibility = empty(len(initial_time)*len(transfer_time), dtype=bool)
optimality = empty(len(initial_time)*len(transfer_time), dtype=bool)
results = empty(len(initial_time)*len(transfer_time), dtype=object)
coords = empty((len(initial_time)*len(transfer_time), 2))
count = 0

initial_control = array([-0.61559512, 0.16343542, -0.07925525, 0.09549926])

for i in range(len(initial_time)):
    for j in range(len(transfer_time)):
        coords[count] = (initial_time[i], transfer_time[j])
        print(count)
        print(coords[count])


        def simulation_3b(control: ndarray) -> ndarray:
            eom = CR3BP(primary=earth, secondary=moon)
            # get initial state and convert to three body frame
            position, velocity, period = get_initial_state(initial_alt)
            position, velocity, period = twobody_to_threebody(position, velocity, period)
            state = concatenate((position, velocity))

            # propagate for initial time
            rk54 = RKF54()
            rk54.add_drift_vector_field(vector_field=eom.evaluate)
            rk54.evaluate(s=state, t0=0., tf=initial_time[i])

            # add initial burn
            position = rk54.states[-1][0:3]
            velocity = rk54.states[-1][3:6]
            velocity[0:2] += control[0:2]
            state = concatenate((position, velocity))

            # propagate transfer ellipse
            rk54 = RKF54()
            rk54.add_drift_vector_field(vector_field=eom.evaluate)
            rk54.evaluate(s=state, t0=0., tf=transfer_time[j])

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
            rk54.evaluate(s=state, t0=0., tf=(initial_time[i] + transfer_time[j]))
            desired_state = rk54.states[-1]
            error = norm((final_state - desired_state) / norm(desired_state))
            # print('three body error: {}'.format(error))
            return error

        def objective_function(control) -> float:
            # print('cost: {}'.format(norm(control)))
            cost = norm(control)
            return cost

        lower_bounds = -max_dv * ones(4, )
        upper_bounds = max_dv * ones(4, )
        bounds = Bounds(lb=lower_bounds, ub=upper_bounds)
        nonlinear_constraint_3b = NonlinearConstraint(final_constraint_3b, -constraint_tol, constraint_tol, jac='2-point')

        result = minimize(objective_function, initial_control, jac='2-point', hess=BFGS(),
                             constraints=[nonlinear_constraint_3b],
                             options={'verbose': 1, 'maxiter': 1000, 'factorization_method': 'SVDFactorization', 'disp': True},
                             bounds=bounds)

        final_solution_history[count] = result.x
        if result.success:
            feasibility[count] = True
            optimality[count] = True
        else:
            if final_constraint_3b(result.x) < constraint_tol:
                feasibility[count] = True
                optimality[count] = False
            else:
                feasibility[count] = False
                optimality[count] = False
        results[count] = result
        count += 1

savez('threebody_BFGS', final_solution_history, coords, feasibility, optimality, results)

