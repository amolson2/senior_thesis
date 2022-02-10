from numpy import savez, reshape, arange, load
from numpy.linalg import norm
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.cm as cm

npzfile = load('0.1_200_0.001_take2.npz')
final_solution_history = npzfile['arr_0']
# first_burn_location = npzfile['arr_1']
# second_burn_location = npzfile['arr_2']
# global_history = npzfile['arr_0']
# final_solution_history = npzfile['arr_1']
first_burn_location = npzfile['arr_1']
second_burn_location = npzfile['arr_2']
num_dimensions = 2

final_solution_history = reshape(final_solution_history, (-1, 2*num_dimensions))
first_burn_location = reshape(first_burn_location, (-1, num_dimensions))
second_burn_location = reshape(second_burn_location, (-1, num_dimensions))
final_solution_history_norm = norm(final_solution_history, axis=1)
first_burn_magnitude = norm(final_solution_history[:, 0:2], axis=1)
second_burn_magnitude = norm(final_solution_history[:, 2:4], axis=1)

# magnitude of first burn vs second burn
plt.figure()
plt.scatter(first_burn_magnitude, second_burn_magnitude)
plt.title('Magnitude of first burn vs. second burn')
plt.xlabel('First burn magnitude')
plt.ylabel('Second burn magnitude')


# scatter plot - final solution history (just optimal solutions)
plt.figure()
x = arange(len(final_solution_history))
plt.plot(final_solution_history_norm, 'bo')
plt.title('Final solution history')
plt.xlabel('Solution number')
plt.ylabel('Total magnitude of control')

# first burn, second burn in plane, total norm on z-axis
plt.figure()
ax = plt.axes(projection='3d')
z = norm(final_solution_history, axis=1)
ax.scatter(first_burn_magnitude, second_burn_magnitude, z, c=z, cmap='viridis', linewidth=0.5)
ax.set_title('First and second burn magnitudes vs. total control magnitude')
ax.set_xlabel('Magnitude of first burn')
ax.set_ylabel('Magnitude of second burn')
ax.set_zlabel('Total magnitude of both burns')

# first burn 3D scatter
plt.figure()
ax = plt.axes(projection='3d')
# z = norm(final_solution_history[:, 0:2], axis=1)
z = arange(len(final_solution_history))
ax.scatter(final_solution_history[:, 0], final_solution_history[:, 1], z, c=z, cmap='viridis', linewidth=0.5)
# ax.plot_trisurf(final_solution_history[:, 0], final_solution_history[:, 1], z, cmap='viridis')
ax.set_title('First burn 3D scatter')
ax.set_xlabel('X component of first burn')
ax.set_ylabel('Y component of first burn')
ax.set_zlabel('magnitude of first burn')

# second burn 3D scatter
plt.figure()
ax = plt.axes(projection='3d')
z = norm(final_solution_history[:, 2:4], axis=1)
ax.scatter(final_solution_history[:, 2], final_solution_history[:, 3], z, c=z, cmap='viridis', linewidth=0.5)
ax.set_title('Second burn 3D scatter')
ax.set_xlabel('X component of second burn')
ax.set_ylabel('Y component of second burn')
ax.set_zlabel('magnitude of second burn')

# colors = norm(final_solution_history[:, 0:2], axis=1)
colors = arange(len(final_solution_history))
normalize = matplotlib.colors.Normalize()
normalize.autoscale(colors)
colormap = cm.viridis
# quiver plot - first burn
plt.figure()
plt.quiver(first_burn_location[:, 0], first_burn_location[:, 1], final_solution_history[:, 0], final_solution_history[:, 1], color=colormap(normalize(colors)))
plt.title('First burn location and direction')
plt.xlabel('X location of first burn (km)')
plt.ylabel('Y location of first burn (km)')

# colors = norm(final_solution_history[:, 2:4], axis=1)
colors = arange(len(final_solution_history))
normalize.autoscale(colors)
# quiver plot - second burn
plt.figure()
plt.quiver(second_burn_location[:, 0], second_burn_location[:, 1], final_solution_history[:, 2], final_solution_history[:, 3], color=colormap(normalize(colors)))
plt.title('Second burn location and direction')
plt.xlabel('X location of second burn (km)')
plt.ylabel('Y location of second burn (km)')

plt.show()