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

class Spacecraft:
    def __init__(self):
        self.solution_history = []
        self.shift_parameter = 0
        self.shift_cost = 0

    def objective_shifted(self, control) -> float:
        j = absolute(norm(control) - self.shift_cost - self.shift_parameter)
        self.solution_history.append(control)
        return j

spacecraft = Spacecraft()
level_count = ones(200).astype(int)
final_solution_history = zeros((200, 4))
first_burn_location = zeros((200, 2))
second_burn_location = zeros((200, 2))
# level_count = ones(10).astype(int)
# level_count[0] = 10
# level_count[1:3] = 2
#
# final_solution_history = zeros((350, 4))
# first_burn_location = zeros((350, 2))
# second_burn_location = zeros((350, 2))
total_count = 0
last_level_solution_history = zeros((1, 4))

for i in range(len(level_count)):
    level_solution_history = zeros((len(last_level_solution_history)*level_count[i], 2*num_dimensions))
    # print('level = {}'.format(i+1))
    for k in range(len(last_level_solution_history)):
        initial_control = last_level_solution_history[k]
        for j in range(level_count[i]):
            perturbation = 0.1*(1-(2*npr.rand(2*num_dimensions)))
            initial_control += perturbation
            result = minimize(spacecraft.objective_shifted, initial_control,
                            method='trust-constr', jac='2-point', hess=BFGS(),
                            constraints=[nonlinear_constraint],
                            options={'verbose': 1},
                            bounds=bounds)
            final_control = result.x
            # print(norm(final_control))
            level_solution_history[j] = final_control
            final_solution_history[total_count] = final_control
            first_burn_location[total_count] = simulation(final_control)[0][0:num_dimensions]
            second_burn_location[total_count] = simulation(final_control)[-1][0:num_dimensions]
            total_count += 1
    last_level_solution_history = level_solution_history
    spacecraft.shift_cost = norm(final_control)
    spacecraft.shift_parameter = 0.001

solution_history = reshape(spacecraft.solution_history, (-1, 4))
savez('first_attempt_backwards', solution_history, final_solution_history, first_burn_location, second_burn_location)

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


plt.show()
