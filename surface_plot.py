import plotly.graph_objects as go

from numpy import load
from numpy.linalg import norm


npzfile = load('surface_plot_data.npz')

final_solution_history = npzfile['arr_0']
coords = npzfile['arr_1']

x = coords[:, 0]
y = coords[:, 1]
z = norm(final_solution_history, axis=1)
fig = go.Figure(data=[go.Mesh3d(x=x, y=y, z=z, opacity=1, intensity=z, colorscale='Viridis')])
fig.show()