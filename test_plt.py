import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def generate_teapot():
    # Define the vertices of the cube
    vertices = np.array([
        [-1, -1, -1],
        [1, -1, -1],
        [1, 1, -1],
        [-1, 1, -1],
        [-1, -1, 1],
        [1, -1, 1],
        [1, 1, 1],
        [-1, 1, 1]
    ])

    # Define the edges of the cube
    edges = [
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 0],
        [4, 5],
        [5, 6],
        [6, 7],
        [7, 4],
        [0, 4],
        [1, 5],
        [2, 6],
        [3, 7]
    ]
    
    return vertices, edges

# Create a figure and a 3D axes
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Generate teapot vertices and edges
vertices, edges = generate_teapot()

# Plot the teapot
for edge in edges:
    ax.plot3D(vertices[edge, 0], vertices[edge, 1], vertices[edge, 2], 'b')

# Set axis labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Set plot title
ax.set_title('3D Teapot')

# Show the plot
plt.show()

