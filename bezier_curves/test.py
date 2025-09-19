import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib.animation import FuncAnimation 
from bezier_curves import bezier_curve, bernstein_poly

control_points = np.array([[0,0], [1,2], [3,3], [4,0]])

num_points = 1000 

curve = bezier_curve(control_points, num_points)

plt.plot(curve[:,0], curve[:,1], 'b-', label="Bezier Curve")

plt.plot(control_points[:,0], control_points[:,1], "ro--", label="Control Points")

plt.legend()
plt.show()
