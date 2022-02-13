from pybeebase.integrator import RKF54
from pybeeastro.bodies import Earth
from pybeeastro.eom.twobody import TwoBody
from pybeeastro.astrodynamics import coe2rv, rv2coe
from pybeebase.plots.lineplot import LinePlot
from numpy import array, concatenate, ndarray, pi, sqrt, ones, zeros, reshape, divide
from numpy.linalg import norm
from scipy.optimize import Bounds, NonlinearConstraint, BFGS, minimize
import matplotlib.pyplot as plt

num_dimensions = 2
solution_history = []
timeshift = 0.2

def generate_initial_orbit():
    earth = Earth()
    classical_orbital_elements = array([earth.equatorial_radius + 2000., 0., 0., 0., 0., 0.])
    period = get_orbital_period(classical_orbital_elements[0], earth.mu_for_children)
    position, velocity = coe2rv(classical_orbital_elements=classical_orbital_elements, num_dimensions=num_dimensions,
                                mu=earth.mu_for_children)
    state = concatenate((position, velocity))
    eom = TwoBody(primary=earth, dimension=num_dimensions)
    rk54 = RKF54()
    rk54.add_drift_vector_field(vector_field=eom.evaluate)
    rk54.evaluate(s=state, t0=0., tf=timeshift*period)
    return rk54.states


def generate_final_orbit():
    earth = Earth()
    classical_orbital_elements = array([earth.equatorial_radius + 3000., 0., 0., 0., 0., 0.])
    period = get_orbital_period(classical_orbital_elements[0], earth.mu_for_children)
    position, velocity = coe2rv(classical_orbital_elements=classical_orbital_elements, num_dimensions=num_dimensions,
                                mu=earth.mu_for_children)
    state = concatenate((position, velocity))
    eom = TwoBody(primary=earth, dimension=num_dimensions)
    rk54 = RKF54()
    rk54.add_drift_vector_field(vector_field=eom.evaluate)
    rk54.evaluate(s=state, t0=0., tf=period)
    return rk54.states


def generate_final_arc(initial_state: ndarray):
    earth = Earth()
    eom = TwoBody(primary=earth, dimension=num_dimensions)
    rk54 = RKF54()
    rk54.add_drift_vector_field(vector_field=eom.evaluate)
    rk54.evaluate(s=initial_state, t0=0., tf=60. * 60.)
    return rk54.states


def get_orbital_period(semimajor: float,
                       mu: float) -> float:
    return 2 * pi * sqrt(((semimajor ** 3) / mu))

def simulation(control: ndarray) -> ndarray:
    # define initial orbit
    earth = Earth()
    classical_orbital_elements = array([earth.equatorial_radius + 2000., 0., 0., 0., 0., 0.])
    position, velocity = coe2rv(classical_orbital_elements=classical_orbital_elements, num_dimensions=num_dimensions,
                                mu=earth.mu_for_children)
    state = concatenate((position, velocity))
    # print(state)

    # propagate orbit for a given input amount of time
    eom = TwoBody(primary=earth, dimension=num_dimensions)
    rk54 = RKF54()
    period = get_orbital_period(classical_orbital_elements[0], earth.mu_for_children)
    rk54.add_drift_vector_field(vector_field=eom.evaluate)
    rk54.evaluate(s=state, t0=0, tf=timeshift*period)

    # add initial burn, propagate for flexible amount of time as a control parameter
    position = rk54.states[-1][0:2]
    velocity = rk54.states[-1][2:4]
    velocity += control[0:num_dimensions]
    # print(rk54.states[-1])
    # rk54.states[-1][2:4] += control[0:2]
    state = concatenate((position, velocity))
    # rk54.add_drift_vector_field(vector_field=eom.evaluate)
    rk54.evaluate(s=state, t0=0, tf=control[-1])

    # add the final burn to the last state
    rk54.states[-1][num_dimensions:2*num_dimensions] += control[num_dimensions:2*num_dimensions]

    return rk54.states


def final_constraint(control) -> float:
    final_state = simulation(control)[-1]
    desired_state = [-9.36975486e+03, -1.72966579e+00,  1.62704991e-03, -6.52257681e+00]
    error = norm((final_state-desired_state)/norm(desired_state))
    return error

def objective_function(control) -> float:
    # print('cost: {}'.format(norm(control)))
    cost = norm(control[0:4])
    return cost


lower_bounds = [-3, -3, -3, -3, 0]
upper_bounds = [3, 3, 3, 3, 81352]
bounds = Bounds(lb=lower_bounds, ub=upper_bounds)

nonlinear_constraint = NonlinearConstraint(final_constraint, -0.01, 0.01, jac='2-point', hess=BFGS())

initial_control = [0, 0, 0, 0, 4000]

result = minimize(objective_function, initial_control,
                  method='trust-constr', jac='2-point', hess=BFGS(),
                  constraints=[nonlinear_constraint],
                  options={'verbose': 1},
                  bounds=bounds)

print(result.x)
print(norm(result.x[0:4]))

plt.figure()
states = generate_initial_orbit()
plt.plot(states[:, 0], states[:, 1], 'r')
states = generate_final_orbit()
plt.plot(states[:, 0], states[:, 1], 'b')
states = simulation(result.x)
plt.plot(states[:, 0], states[:, 1], 'g')

plt.show()