from pybeebase.integrator import RKF54
from pybeeastro.bodies import Earth
from pybeeastro.eom.twobody import TwoBody
from pybeeastro.astrodynamics import coe2rv, rv2coe
from pybeebase.plots.lineplot import LinePlot
from numpy import array, concatenate, ndarray, pi, sqrt, ones, zeros, reshape, datetime64
from numpy.linalg import norm
from scipy.optimize import Bounds, NonlinearConstraint, BFGS, minimize

num_dimensions = 2
solution_history = []

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
    rk54.evaluate(s=state, t0=0., tf=5*period)
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
    # propagate orbit for a given input amount of time
    eom = TwoBody(primary=earth, dimension=num_dimensions)
    rk54 = RKF54()
    rk54.add_drift_vector_field(vector_field=eom.evaluate)
    rk54.evaluate(s=state, t0=0., tf=0)

    # add initial burn, propagate for flexible amount of time as a control parameter
    velocity += control[0:1]
    state = concatenate((position, velocity))
    rk54.add_drift_vector_field(vector_field=eom.evaluate)
    rk54.evaluate(s=state, t0=0., tf=control[4])

    # add the final burn to the last state
    rk54.states[-1][num_dimensions:] += control[2:3]

    return rk54.states


def final_constraint(control) -> float:
    final_state = simulation(control)[-1]
    earth = Earth()
    classical_orbital_elements = array([earth.equatorial_radius + 3000., 0., 0., 0., 0., 0.])
    desired_state = coe2rv(classical_orbital_elements=classical_orbital_elements, num_dimensions=num_dimensions,
                                mu=earth.mu_for_children)
    desired_state = reshape(desired_state, (1, 2*num_dimensions))
    error = (final_state-desired_state)
    if num_dimensions == 3:
        error /= array([earth.equatorial_radius, 1., 1.])
    else:
        # error /= array([earth.equatorial_radius, 1.])
        error = norm(error)
    return error


class Spacecraft:
    def __init__(self):
        self.solution_history = []
        self.cost_history = []

    def objective_function(self, control) -> float:
        # print('cost: {}'.format(norm(control)))
        cost = norm(control)
        # self.cost_history.append(cost)
        # self.solution_history.append(control)
        return cost

lower_bounds = [-3, -3, -3, -3, 0]
upper_bounds = [3, 3, 3, 3, 100000]
bounds = Bounds(lb=lower_bounds, ub=upper_bounds)

nonlinear_constraint = NonlinearConstraint(final_constraint, -0.01, 0.01, jac='2-point', hess=BFGS())

spacecraft = Spacecraft()
initial_control = [-1.45530073e-04,  1.80307396e-01, -1.23255470e-04, -1.75698647e-01, 2000]

result = minimize(spacecraft.objective_function, initial_control,
                method='trust-constr', jac='2-point', hess=BFGS(),
                constraints=[nonlinear_constraint],
                options={'verbose': 1},
                bounds=bounds)
final_control = result.x
deltav = final_control[0:3]
print('deltav: {}'.format(deltav))
time = final_control[4]
print('time: {}'.format(time))


# print(simulation(result.x)[0])
# print(simulation(result.x)[-1])

# solution_history = reshape(solution_history, (-1, 4))
# print('final control: {}'.format(result.x))
# print('first thrust: {}'.format(norm(result.x[:num_dimensions])))
# print('second thrust: {}'.format(norm(result.x[num_dimensions+1:])))
# print('final control norm: {}'.format(norm(result.x[:num_dimensions])+norm(result.x[num_dimensions+1:])))
# print('solution history: {}'.format(solution_history))
# print('history of control norms: {}'.format(norm(solution_history, axis=1)))
# print(solution_history.shape)
# print(norm(solution_history, axis=1).shape)

p = LinePlot()
states = generate_initial_orbit()
p.plot(xdata=states[:, 0], ydata=states[:, 1], color='blue')
states = generate_final_orbit()
p.plot(xdata=states[:, 0], ydata=states[:, 1], color='red')
states = simulation(result.x)
p.plot(xdata=states[:, 0], ydata=states[:, 1], color='black')
states = generate_final_arc(initial_state=states[-1])
p.plot(xdata=states[:, 0], ydata=states[:, 1], color='black', linestyle='--')
p.grid()
p.set_xlabel(xlabel_in=r'$q_0$', fontsize=16)
p.set_ylabel(ylabel_in=r'$q_1$', fontsize=16)

p.new_plot()
states = generate_initial_orbit()
p.plot(xdata=states[:, 0], ydata=states[:, 2], color='blue')
states = generate_final_orbit()
p.plot(xdata=states[:, 0], ydata=states[:, 2], color='red')
states = simulation(result.x)
p.plot(xdata=states[:, 0], ydata=states[:, 2], color='black')
states = generate_final_arc(initial_state=states[-1])
p.plot(xdata=states[:, 0], ydata=states[:, 2], color='black', linestyle='--')
p.grid()
p.set_xlabel(xlabel_in=r'$q_0$', fontsize=16)
p.set_ylabel(ylabel_in=r'$q_2$', fontsize=16)


p.show()