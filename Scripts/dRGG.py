import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# Function to randomly position nodes in the unit square
def generate_random_positions(num_nodes):
    positions = {i: (np.random.uniform(0, 1), np.random.uniform(0, 1)) for i in range(num_nodes)}
    return positions
    
# Function to define connection radius based on arbitrary parameters
def connection_radius(low, a):
	if a == 1.0:
		eta = np.log( np.sqrt(2.0)/low )
		eta = 1.0/eta
		U  = np.random.uniform(0,1)
		radiusL = np.exp(U/eta)
	else:
		expo = a - 1.0
		eta = expo*(low**expo)*(  (1.0 - (low/np.sqrt(2.0))**(expo) )**(-1.0)  )
		U =np.random.uniform(0,1)
		expo = a - 1.0
		radiusL = (  (eta - expo*U*(low**(expo)) ) / (eta*(low**(expo)))  )**( -1.0/expo)
	return radiusL
		
	
low = 0.1
a = 4
# Create random positions for nodes
num_nodes = 45
positions = generate_random_positions(num_nodes)
radius = np.zeros(num_nodes)
# Create the graph and add edges based on the connection radius
G = nx.DiGraph()
G.add_nodes_from(positions.keys())

for i in range(num_nodes):
	radius[i] = connection_radius(low, a)
	#print(i,radius[i])

for i in positions:
	for j in positions:
		if j!=i:
			x_i, y_i = positions[i]
			x_j, y_j = positions[j]
			distance = np.sqrt((x_i - x_j)**2 + (y_i - y_j)**2)
			if distance <= radius[i]:
				G.add_edge(i, j)

# Plot the graph
fig, ax = plt.subplots(figsize=(8, 8))
plt.gca().set_aspect('equal', adjustable='box')
plt.xlim(0, 1)
plt.ylim(0, 1)

# Draw the graph
#nx.draw(G, pos=positions, with_labels=False, node_size=150, node_color='blue', edge_color='gray')
# Draw the edges and nodes of the graph
#nx.draw(G, pos=positions, ax=ax, node_color='skyblue', edge_color='gray', node_size=150, with_labels=False)
# ~ nx.draw(
    # ~ G,
    # ~ pos=positions,
    # ~ ax=ax,
    # ~ node_color='skyblue',
    # ~ edge_color='gray',
    # ~ node_size=150,
    # ~ with_labels=False,
    # ~ arrows=True,  # Add arrows to visualize direction
    # ~ arrowsize=10,  # Customize arrow size
    
    # ~ )

# Draw the enclosing square from (0,0) to (1,1)
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.gca().set_aspect('equal', adjustable='box')
plt.gca().add_patch(plt.Rectangle((0, 0), 1, 1, edgecolor='black', fill=False, linewidth=1))

# Highlight the connection radius for a specific node
highlight_node = 10
highlight_node1 = 20
highlight_node2 = 44
highlight_pos = positions[highlight_node]
highlight_pos1 = positions[highlight_node1]
highlight_pos2 = positions[highlight_node2]




edges_to_highlight = [(u, v) for u, v in G.edges if u == highlight_node]


edges_to_highlight1 = [(u, v) for u, v in G.edges if u == highlight_node1 ]

edges_to_highlight2 = [(u, v) for u, v in G.edges if u == highlight_node2]


# Get all edges except the ones to highlight
all_edges = list(G.edges)
all_edges_to_highlight = edges_to_highlight + edges_to_highlight1 + edges_to_highlight2
non_highlighted_edges = [e for e in all_edges if e not in (all_edges_to_highlight ) ]

# Draw non-highlighted edges in gray

nx.draw_networkx_nodes(G,pos=positions,ax=ax, node_color='skyblue', node_size=150)
nx.draw_networkx_edges(G, pos=positions, edgelist=non_highlighted_edges, edge_color='gray', width=0.5)

nx.draw_networkx_edges(
    G,
    pos=positions,
    edgelist=edges_to_highlight,
    edge_color='red',
    width=1.0, arrows=True)
nx.draw_networkx_edges(
    G,
    pos=positions,
    edgelist=edges_to_highlight1,
    edge_color='blue',
    width=1.0, arrows=True)
nx.draw_networkx_edges(
    G,
    pos=positions,
    edgelist=edges_to_highlight2,
    edge_color='green',
    width=1.0, arrows=True)
    
# Highlight the selected node
ax.scatter(*highlight_pos, color='red', s=150, zorder=2)
ax.scatter(*highlight_pos1, color='blue', s=150, zorder=2)
ax.scatter(*highlight_pos2, color='green', s=150, zorder=2)

# Draw the circle around the highlighted node
circle = plt.Circle(highlight_pos, radius[highlight_node], color='red', fill=False, linestyle='--', linewidth=2)
ax.add_patch(circle)
circle1 = plt.Circle(highlight_pos1, radius[highlight_node1], color='blue', fill=False, linestyle='--', linewidth=2)
ax.add_patch(circle1)
circle2 = plt.Circle(highlight_pos2, radius[highlight_node2], color='green', fill=False, linestyle='--', linewidth=2)
ax.add_patch(circle2)







#plt.title("Random Geometric Graph with Arbitrary Connection Radius")
plt.savefig("dRGGExample.eps", format='eps')
plt.show()
