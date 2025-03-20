import networkx as nx
import matplotlib.pyplot as plt
#from matplotlib.animation import FuncAnimation, FFMpegWriter  <-----FFM... is for MP4
from matplotlib.animation import FuncAnimation, PillowWriter   #for GIF
import numpy as np
import random
import scipy

# Number of frames in the animation
values = 20
num_nodes = 50

# Create the figure and the axis
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

# ~ # Create the figure and the axis
# ~ fig, ax = plt.subplots()

# Function to initialize the graph
def init():
    ax1.set_title("RGG Network Graph Animation")
    ax1.set_xlim(0, 1)  # Set x-axis limits for the unit square
    ax1.set_ylim(0, 1)  # Set y-axis limits for the unit square
    ax1.add_patch(plt.Rectangle((0, 0), 1, 0.9999, edgecolor='black', facecolor='none'))  # Draw the unit square
    ax2.set_title("Adjacency Matrix")
    ax2.axis('on')  # Turn off axis for the adjacency matrix
    ax2.imshow(np.zeros((num_nodes, num_nodes)), cmap='Blues', vmin=0, vmax=1, aspect='equal')  # Initialize the adjacency matrix
   
def radii(frame):
	cont = int(frame)
	radii = np.zeros((values,2))
	radii = np.loadtxt('Proportions_for_Generator_radius.dat')
	radius = radii[cont,1]
	
	return radius
	



# Function to update the graph for each frame
def update(frame):
	ax1.clear()
	L = np.sqrt(2.0)
	pp = frame * (1.0/(values-1))
	cont = 0
	#print(pp)
	radius = radii(frame)
	
	print(radius)
	cont +=1
	# Custom node styles
	node_color = 'blue'
	node_size = 20
	# Custom edge styles
	edge_color = 'gray'
	style = 'solid'  # solid, dashed, dotted, etc.
	width = 0.15
	G = nx.random_geometric_graph(num_nodes, radius, seed = 1000)
	# position is stored as node attribute data for random_geometric_graph
	pos = nx.get_node_attributes(G, "pos")
	# find node near center (0.5,0.5)
	#dmin = 1
	#ncenter = 0
	# ~ for n in pos:
		# ~ x, y = pos[n]
		# ~ d = (x - 0.5) ** 2 + (y - 0.5) ** 2
		# ~ if d < dmin:
			# ~ ncenter = n
			# ~ dmin = d
	# ~ # color by path length from node near center
	#p = dict(nx.single_source_shortest_path_length(G, ncenter))
	#nx.draw_networkx_edges(G, pos, alpha=0.2)
	#nx.draw_networkx_nodes(G,pos,nodelist=list(p.keys()),node_size=80,node_color=node_color, labels=True)#list(p.values())),cmap=plt.cm.Reds_r)
	# Draw the unit square
	ax1.add_patch(plt.Rectangle((0, 0), 1.0, 1.0, edgecolor='black', facecolor='white')) 
	nx.draw(G, pos = pos,node_color=node_color, node_size = node_size, width=width,with_labels=False, ax=ax1)
	ax1.set_title(f"$\\alpha$ = {pp:.2f}", fontsize= 24)
	# Update the adjacency matrix
	adjacency_matrix_sparse = nx.adjacency_matrix(G)
	#adjacency_matrix = nx.adjacency_matrix(G)
	adjacency_matrix = adjacency_matrix_sparse.toarray()
	np.fill_diagonal(adjacency_matrix, 1)
	ax2.clear()
	
	ax2.set_title("Adjacency Matrix",fontsize=24)
	
	ax2.axis('on')
	ax2.imshow(adjacency_matrix, cmap='Blues', vmin=0, vmax=1, aspect='equal', alpha=1)
	plt.tight_layout()
	


    
# ~ def update_adjacency(frame):
	# ~ ax.clear()
	# ~ p = frame*0.1
	# ~ G = nx.erdos_renyi_graph(10,p)


# Create the animation
ani = FuncAnimation(fig, update, frames=range(values), init_func=init, repeat=False)

# Save the animation
#Writer = FFMpegWriter(fps=2, metadata=dict(artist='Me'), bitrate=1800) for MP4
#ani.save("network_graph_animation.mp4", writer=Writer)
ani.save("RGG_network_graph_animation.gif", writer=PillowWriter(fps=0.01))


# Optionally, display the animation
plt.show()
