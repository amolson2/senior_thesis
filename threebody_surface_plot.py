from pybeeastro.bodies import Earth, Moon
from pybeeastro.eom.cr3bp import CR3BP
from numpy import concatenate, pi, sqrt, load, empty
from numpy.linalg import norm
from plotly import graph_objects as go

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


x = feasible_coords[:, 0]
y = feasible_coords[:, 1]
z = norm(feasible_solutions[:, 0:4], axis=1)
fig = go.Figure(data=go.Contour(z=z, x=x, y=y, colorscale='Viridis'))
fig.update_layout(
    title={'text': "Contour Plot of Solution Space for Two Body Time Parameterized Impulsive Transfer",
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'},
    xaxis_title="Initial Time [seconds]",
    yaxis_title="Transfer Time [seconds]",
    legend_title="Cost [normalized velocity units]",
    font=dict(size=18)
)
fig.show()
