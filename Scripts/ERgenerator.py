import networkx as nx
import matplotlib.pyplot as plt
#from matplotlib.animation import FuncAnimation, FFMpegWriter  <-----FFM... is for MP4
from matplotlib.animation import FuncAnimation, PillowWriter   #for GIF
import numpy as np

# Number of frames in the animation
values = 20
num_nodes = 50

# Create the figure and the axis
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

# Function to initialize the graph
def init():
    ax1.set_title("ER Network Graph Animation")
    ax1.set_xlim(0, 1)  # Set x-axis limits for the unit square
    ax1.set_ylim(0, 1)  # Set y-axis limits for the unit square
    ax1.add_patch(plt.Rectangle((0, 0), 1, 1, edgecolor='black', facecolor='none'))  # Draw the unit square
    ax2.set_title("Adjacency Matrix")
    ax2.axis('on')  # Turn off axis for the adjacency matrix
    ax2.imshow(np.zeros((num_nodes, num_nodes)), cmap='Blues', vmin=0, vmax=1, aspect='equal')  # Initialize the adjacency matrix
    

# Function to update the graph for each frame
def update(frame):
    ax1.clear()
    pp = frame * (1.0/(values-1))
    #np.savetxt('Proportions_for_Generator.dat', pp, fmt='%.18e', delimiter=' ', newline='\n')
    print(pp)
    # Custom node styles
    node_color = 'blue'
    node_size = 20
    # Custom edge styles
    edge_color = 'gray'
    style = 'solid'  # solid, dashed, dotted, etc.
    width = 0.15
    G = nx.erdos_renyi_graph(num_nodes, pp, seed = 0)
    
    nx.draw(G,node_color=node_color, node_size = node_size, width= width, with_labels=False, ax=ax1)
    ax1.set_title(f"p = {pp:.2f}",fontsize= 24)
    # Update the adjacency matrix
    #adjacency_matrix = nx.adjacency_matrix(G)
    adjacency_matrix_sparse = nx.adjacency_matrix(G)
    #adjacency_matrix = nx.to_numpy_array(G)
    adjacency_matrix = adjacency_matrix_sparse.toarray()
    np.fill_diagonal(adjacency_matrix, 1)
    ax2.clear()
    ax2.axis('on')
    ax2.set_title("Adjacency Matrix",fontsize=24)
    
    ax2.imshow(adjacency_matrix, cmap='Blues', vmin=0, vmax=1, aspect='equal', alpha=1)
    plt.tight_layout()


# Create the animation
ani = FuncAnimation(fig, update, frames=range(values), init_func=init, repeat=False)

# Save the animation
#Writer = FFMpegWriter(fps=2, metadata=dict(artist='Me'), bitrate=1800) for MP4
#ani.save("network_graph_animation.mp4", writer=Writer)
ani.save("ER_network_graph_animation.gif", writer=PillowWriter(fps=2))


# Optionally, display the animation
plt.show()

