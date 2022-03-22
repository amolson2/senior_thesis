from pybeebase.integrator import RKF54
from pybeeastro.bodies import Earth, Moon
# from pybeeastro.eom.pcr3bp import PCR3BP
from pybeeastro.eom.cr3bp import CR3BP
from pybeeastro.eom.twobody import TwoBody
from pybeeastro.astrodynamics import coe2rv, rv2coe, convert_state_from_primary_centered_2BP_to_CR3BP
from numpy import array, concatenate, ndarray, pi, sqrt, ones
from numpy.linalg import norm
from scipy.optimize import Bounds, NonlinearConstraint, BFGS, minimize
import matplotlib.pyplot as plt

num_dimensions = 2

earth = Earth()
moon = Moon()
initial_alt = 200000.
final_alt = 250000.
# transfer_time = 132856.59730501875/2


constraint_tol = 0.01
max_dv = 3.


def get_orbital_period(semimajor: float,
                       mu: float) -> float:
    return 2 * pi * sqrt(((semimajor ** 3) / mu))


def get_initial_state(alt):
    classical_orbital_elements = array([earth.equatorial_radius + alt, 0., 0., 0., 0., 0.])
    position, velocity = coe2rv(classical_orbital_elements=classical_orbital_elements, mu=earth.mu_for_children)
    semimajor = classical_orbital_elements[0]
    period = 2 * pi * sqrt(((semimajor ** 3) / earth.mu_for_children))
    return position, velocity, period


initial_period = get_orbital_period(earth.equatorial_radius+initial_alt, earth.mu_for_children)
hohmann_semimajor = ((2*earth.equatorial_radius)+initial_alt+final_alt)/2
hohmann_period = get_orbital_period(hohmann_semimajor, earth.mu_for_children)/2
initial_time = 2*initial_period
transfer_time = 3*hohmann_period

def twobody_to_threebody(position, velocity, period):
    state = concatenate((position, velocity))
    eom = CR3BP(primary=earth, secondary=moon)
    period /= eom.normalized_units['TimeUnit']
    transformed_state = convert_state_from_primary_centered_2BP_to_CR3BP(state, earth, moon)
    return transformed_state[0:3], transformed_state[3:6], period


def simulation_2b(control: ndarray) -> ndarray:
    # propagate initial orbit for initial time
    position, velocity, period = get_initial_state(initial_alt)
    state = concatenate((position, velocity))
    eom = TwoBody(primary=earth, dimension=num_dimensions)
    rk54 = RKF54()
    rk54.add_drift_vector_field(vector_field=eom.evaluate)
    rk54.evaluate(s=state, t0=0., tf=initial_time)

    # add initial burn
    position = rk54.states[-1][0:3]
    velocity = rk54.states[-1][3:6]
    velocity[0:2] += control[0:2]
    state = concatenate((position, velocity))

    # propagate transfer ellipse
    eom = TwoBody(primary=earth, dimension=num_dimensions)
    rk54 = RKF54()
    rk54.add_drift_vector_field(vector_field=eom.evaluate)
    rk54.evaluate(s=state, t0=0., tf=transfer_time)

    # add the final burn to the last state
    rk54.states[-1][3:5] += control[2:4]
    return rk54.states


def final_constraint_2b(control) -> float:
    # final state from impulsive transfer simulation
    final_state = simulation_2b(control)[-1]

    # generate desired state by propagating final orbit for half of a period
    position, velocity, period = get_initial_state(final_alt)
    state = concatenate((position, velocity))
    eom = TwoBody(primary=earth, dimension=num_dimensions)
    rk54 = RKF54()
    rk54.add_drift_vector_field(vector_field=eom.evaluate)
    rk54.evaluate(s=state, t0=0., tf=period/2.)
    desired_state = rk54.states[-1]
    error = norm((final_state-desired_state)/norm(desired_state))
    print('two body error: {}'.format(error))
    return error


def simulation_3b(control: ndarray) -> ndarray:
    eom = CR3BP(primary=earth, secondary=moon)
    # get initial state and convert to three body frame
    position, velocity, period = get_initial_state(initial_alt)
    position, velocity, period = twobody_to_threebody(position, velocity, period)
    state = concatenate((position, velocity))

    # propagate for initial time
    rk54 = RKF54()
    rk54.add_drift_vector_field(vector_field=eom.evaluate)
    rk54.evaluate(s=state, t0=0., tf=initial_time/eom.normalized_units['TimeUnit'])

    # add initial burn
    position = rk54.states[-1][0:3]
    velocity = rk54.states[-1][3:6]
    velocity[0:2] += control[0:2]
    state = concatenate((position, velocity))

    # propagate transfer ellipse
    rk54 = RKF54()
    rk54.add_drift_vector_field(vector_field=eom.evaluate)
    rk54.evaluate(s=state, t0=0., tf=transfer_time/eom.normalized_units['TimeUnit'])

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
    rk54.evaluate(s=state, t0=0., tf=period/2.)
    desired_state = rk54.states[-1]
    error = norm((final_state - desired_state) / norm(desired_state))
    print('three body error: {}'.format(error))
    return error


def objective_function(control) -> float:
    # print('cost: {}'.format(norm(control)))
    cost = norm(control)
    return cost


lower_bounds = -max_dv * ones(4,)
upper_bounds = max_dv * ones(4,)
bounds = Bounds(lb=lower_bounds, ub=upper_bounds)

nonlinear_constraint_2b = NonlinearConstraint(final_constraint_2b, -constraint_tol, constraint_tol, jac='2-point', hess=BFGS())
nonlinear_constraint_3b = NonlinearConstraint(final_constraint_3b, -constraint_tol, constraint_tol, jac='2-point', hess=BFGS())

# two body version of optimization
# initial_control = [-1.45530073e-04,  1.80307396e-01, -1.23255470e-04, -1.75698647e-01]
# result_2b = minimize(objective_function, initial_control,
#                     method='trust-constr', jac='2-point', hess=BFGS(),
#                     constraints=[nonlinear_constraint_2b],
#                     options={'verbose': 1, 'maxiter': 1000},
#                     bounds=bounds)
# print(result_2b.x)
# print(norm(result_2b.x[0:4]))


# three body version
# initial_control = result_2b.x
# initial_control = array([1.23136730e-02, 2.42849188e-01, 2.42287465e-08, -3.00501047e-09])
# initial_control = [0, 0, 0, 0]
# result_3b = minimize(objective_function, initial_control,
#                     method='trust-constr', jac='2-point', hess=BFGS(),
#                     constraints=[nonlinear_constraint_3b],
#                     options={'verbose': 1, 'maxiter': 1000},
#                     bounds=bounds)
# print(result_3b.x)
# print(norm(result_3b.x[0:4]))

def generate_orbit_2b(alt):
    position, velocity, period = get_initial_state(alt)
    # propagate for one orbit
    state = concatenate((position, velocity))
    eom = TwoBody(primary=earth, dimension=num_dimensions)
    rk54 = RKF54()
    rk54.add_drift_vector_field(vector_field=eom.evaluate)
    rk54.evaluate(s=state, t0=0., tf=period)
    return rk54.states


def generate_orbit_3b(alt, time):
    # determine orbit in two body
    position, velocity, period = get_initial_state(alt)
    # convert to three body
    position, velocity, period = twobody_to_threebody(position, velocity, period)
    # propagate for one orbit
    state = concatenate((position, velocity))
    eom = CR3BP(primary=earth, secondary=moon)
    rk54 = RKF54()
    rk54.add_drift_vector_field(vector_field=eom.evaluate)
    rk54.evaluate(s=state, t0=0., tf=period*time)
    return rk54.states


# plt.figure()
# states = generate_orbit_2b(initial_alt)
# plt.plot(states[:, 0], states[:, 1], 'r')
# states = generate_orbit_2b(final_alt)
# plt.plot(states[:, 0], states[:, 1], 'b')
# states = simulation_2b(result_2b.x)
# plt.plot(states[:, 0], states[:, 1], 'g')

# result = [-0.03403575, -0.04577415, -0.00303472, -0.04726715]
plt.figure()
states = generate_orbit_3b(initial_alt, 1)
plt.plot(states[:, 0], states[:, 1], 'r')
states = generate_orbit_3b(final_alt, 0.5)
plt.plot(states[:, 0], states[:, 1], 'm')
states = generate_orbit_3b(final_alt, 1)
plt.plot(states[:, 0], states[:, 1], ':b')
states = simulation_3b(result)
plt.plot(states[:, 0], states[:, 1], 'g')
plt.show()