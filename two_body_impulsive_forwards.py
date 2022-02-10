from pybeebase.integrator import RKF54
from pybeeastro.bodies import Earth
from pybeeastro.eom.twobody import TwoBody
from pybeeastro.astrodynamics import coe2rv, rv2coe
from numpy import array, concatenate, ndarray, pi, sqrt, ones, zeros, savez, absolute, reshape, arange
import numpy.random as npr
from numpy.linalg import norm
from scipy.optimize import Bounds, NonlinearConstraint, BFGS, minimize
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.cm as cm
num_dimensions = 2

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
    rk54.evaluate(s=state, t0=0., tf=period)
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
    earth = Earth()
    classical_orbital_elements = array([earth.equatorial_radius + 2000., 0., 0., 0., 0., 0.])
    position, velocity = coe2rv(classical_orbital_elements=classical_orbital_elements, num_dimensions=num_dimensions,
                                mu=earth.mu_for_children)
    # add initial burn
    velocity += control[:num_dimensions]
    state = concatenate((position, velocity))
    # get new classical orbital elements
    classical_orbital_elements = rv2coe(position, velocity, num_dimensions=num_dimensions, mu=earth.mu_for_children)
    semimajor = classical_orbital_elements[0]
    period = get_orbital_period(semimajor, earth.mu_for_children)
    # advect for half the new orbital period
    eom = TwoBody(primary=earth, dimension=num_dimensions)
    rk54 = RKF54()
    rk54.add_drift_vector_field(vector_field=eom.evaluate)
    rk54.evaluate(s=state, t0=0., tf=period / 2.)
    # add the final burn to the last state
    rk54.states[-1][num_dimensions:] += control[num_dimensions:]
    return rk54.states


def final_constraint(control) -> float:
    final_state = simulation(control)[-1]
    earth = Earth()
    classical_orbital_elements = rv2coe(final_state[:num_dimensions], final_state[num_dimensions:],
                                        num_dimensions=num_dimensions, mu=earth.mu_for_children)
    desired_elements = array([earth.equatorial_radius + 3000., 0., 0., 0., 0., 0.])
    error = (classical_orbital_elements - desired_elements)[:num_dimensions]
    if num_dimensions == 3:
        error /= array([earth.equatorial_radius, 1., 1.])
    else:
        error /= array([earth.equatorial_radius, 1.])
        error = norm(error)
    return error

lower_bounds = -3. * ones(num_dimensions*2,)
upper_bounds = 3. * ones(num_dimensions*2,)
bounds = Bounds(lb=lower_bounds, ub=upper_bounds)

nonlinear_constraint = NonlinearConstraint(final_constraint, -0.01, 0.01, jac='2-point', hess=BFGS())

initial_control = [-1.45530073e-04,  1.80307396e-01, -1.23255470e-04, -1.75698647e-01]
# initial_control = array([0., 1., 0., 0., -1., 0.])
# initial_control = array([-1.45498296e-04, 1.80307395e-01, -1.23270635e-04, 3.69840525e-09])

class Spacecraft:
    def __init__(self):
        self.solution_history = []
        self.cost_history = []

    def objective_function(self, control) -> float:
        # print('cost: {}'.format(norm(control)))
        cost = norm(control)
        if cost < 0.3:
            self.solution_history.append(control)
        return cost

spacecraft = Spacecraft()
first_burn_location = []
second_burn_location = []
result = minimize(spacecraft.objective_function, initial_control,
                  method='trust-constr', jac='2-point', hess=BFGS(),
                  constraints=[nonlinear_constraint],
                  options={'verbose': 1},
                  bounds=bounds)

final_control = result.x
# spacecraft.solution_history.append(final_control)
first_burn_location.append(simulation(final_control)[0][0:num_dimensions])
second_burn_location.append(simulation(final_control)[-1][0:num_dimensions])
num_perturbations = 100
delta_x1 = 0.1*(1-(2*npr.rand(num_perturbations)))
delta_y1 = 0.1*(1-(2*npr.rand(num_perturbations)))
delta_x2 = 0.1*(1-(2*npr.rand(num_perturbations)))
delta_y2 = 0.1*(1-(2*npr.rand(num_perturbations)))
perturbations = [final_control[0]+delta_x1, final_control[1]+delta_y1, final_control[2]+delta_x2, final_control[3]+delta_y2]
perturbations = reshape(perturbations, (num_perturbations, 4))

for i in range(num_perturbations):
    result = minimize(spacecraft.objective_function, perturbations[i,:],
                    method='trust-constr', jac='2-point', hess=BFGS(),
                    constraints=[nonlinear_constraint],
                    options={'verbose': 1},
                    bounds=bounds)
    final_control = result.x
    # spacecraft.solution_history.append(final_control)
    first_burn_location.append(simulation(final_control)[0][0:num_dimensions])
    second_burn_location.append(simulation(final_control)[-1][0:num_dimensions])

solution_history = reshape(spacecraft.solution_history, (-1, 4))
solution_history_norm = norm(solution_history, axis=1)
first_burn_magnitude = norm(solution_history[:, 0:2], axis=1)
second_burn_magnitude = norm(solution_history[:, 2:4], axis=1)

plt.figure()
plt.scatter(first_burn_magnitude, second_burn_magnitude)
plt.title('Magnitude of first burn vs. second burn')

plt.figure()
plt.plot(solution_history_norm, 'bo')
plt.title('Final solution history')
plt.xlabel('Solution number')
plt.ylabel('Total magnitude of control')

# first burn 3D scatter
plt.figure()
ax = plt.axes(projection='3d')
z = norm(solution_history[:, 0:2], axis=1)
ax.scatter(solution_history[:, 0], solution_history[:, 1], z, c=z, cmap='viridis', linewidth=0.5)
plt.title('first burn 3D scatter')

# second burn 3D scatter
plt.figure()
ax = plt.axes(projection='3d')
z = norm(solution_history[:, 2:4], axis=1)
ax.scatter(solution_history[:, 2], solution_history[:, 3], z, c=z, cmap='viridis', linewidth=0.5)
plt.title('second burn 3D scatter')

# two burns

plt.show()