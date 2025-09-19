import numpy as np 
import bezier 
import matplotlib.pyplot as plt 
from matplotlib.animation import FuncAnimation 

curves_nodes = [
    np.asfortranarray([[0.0, 1.0, 2.0, 3.0],
                       [0.0, 3.0, 3.0, 0.0]]),
    np.asfortranarray([[3.0, 4.0, 5.0, 6.0],
                       [0.0, -2.0, 2.0, 0.0]]),
    np.asfortranarray([[6.0, 7.0, 8.0, 9.0],
                       [0.0, 3.0, 3.0, 0.0]])
]

curves = [bezier.Curve(nodes, degree=3) for nodes in curves_nodes] 

num_points = 200 
s_vals = np.linspace(0.0, 1.0, num_points) 
all_points = [curve.evaluate_multi(s_vals) for curve in curves] 

fig, ax = plt.subplots() 
ax.set_xlim(-1, 10) 
ax.set_ylim(-3, 4) 

lines = [ax.plot([], [], lw=2)[0] for _ in curves] 

def update(frame): 
    for i, points in enumerate(all_points): 
        lines[i].set_data(points[0, :frame], points[1, :frame])
    return lines 

ani = FuncAnimation(fig, update, frames=num_points, blit=True, interval=30, repeat=True) 

plt.show()
 
