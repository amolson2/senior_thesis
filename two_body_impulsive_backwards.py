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
npr.seed(10)

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
num_shifts = 200

class Spacecraft:
    def __init__(self):
        self.solution_history = []
        self.shift_parameter = 0
        self.shift_cost = 0

    def objective_shifted(self, control) -> float:
        j = absolute(norm(control) - self.shift_cost - self.shift_parameter)
        # self.solution_history.append(control)
        return j

spacecraft = Spacecraft()
initial_control = zeros(2*num_dimensions)
final_solution_history = zeros((num_shifts, 4))
first_burn_location = zeros((num_shifts, 2))
second_burn_location = zeros((num_shifts, 2))

for i in range(num_shifts):
    perturbation = 0.1*(1-(2*npr.rand(2*num_dimensions)))
    initial_control += perturbation
    result = minimize(spacecraft.objective_shifted, initial_control,
                    method='trust-constr', jac='2-point', hess=BFGS(),
                    constraints=[nonlinear_constraint],
                    options={'verbose': 1},
                    bounds=bounds)
    final_control = result.x
    first_burn_location[i] = simulation(final_control)[0][0:num_dimensions]
    second_burn_location[i] = simulation(final_control)[-1][0:num_dimensions]
    final_solution_history[i] = final_control
    initial_control = final_control
    spacecraft.shift_parameter = 0.001
    spacecraft.shift_cost = norm(final_control)

savez('0.1_200_0.001_take2', final_solution_history, first_burn_location, second_burn_location)
final_solution_history_norm = norm(final_solution_history, axis=1)
first_burn_magnitude = norm(final_solution_history[:, 0:2], axis=1)
second_burn_magnitude = norm(final_solution_history[:, 2:4], axis=1)

# magnitude of first burn vs second burn
plt.figure()
plt.scatter(first_burn_magnitude, second_burn_magnitude)
plt.title('Magnitude of first burn vs. second burn')


# scatter plot - final solution history (just optimal solutions)
plt.figure()
x = arange(len(final_solution_history))
plt.scatter(x, norm(final_solution_history, axis=1))
plt.title('Final solution history (just optimal solutions)')

# first burn, second burn in plane, total norm on z-axis
plt.figure()
ax = plt.axes(projection='3d')
z = norm(final_solution_history, axis=1)
ax.scatter(first_burn_magnitude, second_burn_magnitude, z, c=z, cmap='viridis', linewidth=0.5)
plt.title('First and second burn magnitudes vs. total control magnitude')

# first burn 3D scatter
plt.figure()
ax = plt.axes(projection='3d')
z = norm(final_solution_history[:, 0:2], axis=1)
ax.scatter(final_solution_history[:, 0], final_solution_history[:, 1], z, c=z, cmap='viridis', linewidth=0.5)
plt.title('first burn 3D scatter')

# second burn 3D scatter
plt.figure()
ax = plt.axes(projection='3d')
z = norm(final_solution_history[:, 2:4], axis=1)
ax.scatter(final_solution_history[:, 2], final_solution_history[:, 3], z, c=z, cmap='viridis', linewidth=0.5)
plt.title('second burn 3D scatter')


colors = norm(final_solution_history[:, 0:2], axis=1)
normalize = matplotlib.colors.Normalize()
normalize.autoscale(colors)
colormap = cm.viridis
# quiver plot - first burn
plt.figure()
plt.quiver(first_burn_location[:, 0], first_burn_location[:, 1], final_solution_history[:, 0], final_solution_history[:, 1], color=colormap(normalize(colors)))
plt.title('first burn quiver')
plt.xlabel('X location of first burn (km)')
plt.ylabel('Y location of first burn (km')

colors = norm(final_solution_history[:, 2:4], axis=1)
normalize.autoscale(colors)
# quiver plot - second burn
plt.figure()
plt.quiver(second_burn_location[:, 0], second_burn_location[:, 1], final_solution_history[:, 2], final_solution_history[:, 3], color=colormap(normalize(colors)))
plt.title('second burn quiver')
plt.xlabel('X location of second burn (km)')
plt.ylabel('Y location of second burn (km')

plt.show()
