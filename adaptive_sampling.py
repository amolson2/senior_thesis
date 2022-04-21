from pybeebase.integrator import RKF54
from pybeeastro.bodies import Earth, Moon
from pybeeastro.eom.cr3bp import CR3BP
from pybeeastro.astrodynamics import coe2rv, convert_state_from_primary_centered_2BP_to_CR3BP
from numpy import array, concatenate, ndarray, pi, sqrt, load, reshape
from numpy.linalg import norm
from numpy.random import rand, uniform
from scipy.optimize import Bounds, NonlinearConstraint, BFGS, minimize
from sklearn.cluster import KMeans
from time import process_time
import plotly.express as px


initial_start = process_time()
num_clusters = 22
npzfile = load('verification_results_300000_0.001.npz')
solutions = npzfile['arr_0']
kmeans = KMeans(n_clusters=num_clusters, random_state=10).fit(solutions)
fig = px.scatter(x=solutions[:, -2], y=solutions[:, -1], color=kmeans.labels_)
fig.show()

clusters_discovered = []

max_dv = 1.
max_time = 8
constraint_tol = 0.0001
num_dimensions = 3
earth = Earth()
moon = Moon()
initial_alt = 200000.
final_alt = 300000.


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
upper_bounds = [max_dv, max_dv, max_dv, max_dv, 10, 10]
bounds = Bounds(lb=lower_bounds, ub=upper_bounds)
nonlinear_constraint_3b = NonlinearConstraint(final_constraint_3b, -constraint_tol, constraint_tol, jac='2-point')


def objective_function(control) -> float:
    # print('cost: {}'.format(norm(control)))
    cost = norm(control[0:4])
    return cost

optimizing_time = 0
comparison_time = 0

while len(clusters_discovered) < num_clusters:
    dv = uniform(low=-max_dv, high=max_dv, size=(4,))
    times = max_time*rand(2)
    initial_control = concatenate((dv, times))
    print(initial_control)
    label = kmeans.predict(reshape(initial_control, (1, 6)))
    print(label)
    if label not in clusters_discovered:
        start = process_time()
        result = minimize(objective_function, initial_control, jac='2-point',
                                constraints=[nonlinear_constraint_3b],
                                options={'maxiter': 500, 'disp': True},
                                bounds=bounds)

        end = process_time()
        optimizing_time += (end-start)
        print(result.x)
        print(optimizing_time)
        start = process_time()
        if result.success:
            label = kmeans.predict(reshape(result.x, (1, 6)))
            print(label)
            if label not in clusters_discovered:
                clusters_discovered.append(label)
                print(clusters_discovered)
        end = process_time()
        comparison_time += (end-start)
        print(comparison_time)

final_stop = process_time()
print(final_stop)