import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random as rd
from scipy.linalg import eigh


class RandomNetworkModels:
	def __init__(self, num_nodes: int, model: str = "ER", **kwargs):
		"""
		Initialize the network with a given model.

		Parameters:
		- num_nodes: The total number of nodes in the network
		- model: Type of random network ('ER' for Erdős-Rényi,
		'RGG' for Random Geometric Graph, and
		'HRG' for Hyperbolic random graph)
		- kwargs: Additional parameters for network generation
		Methods:
		- generate_network: generates the network based on the selected model.
		- adjacency_matrix_er: two numpy arrays (dim, dim), dim = num_nodes. A weighted and a binary adjacency matrix of the ER network.
		- adjacency_matrix_rgg: two numpy arrays (dim, dim), dim = num_nodes. A weighted and a binary adjacency matrix of the RGG network.
		- adjacency_matrix_hrg: two numpy arrays (dim, dim), dim = num_nodes. A weighted and a binary adjacency matrix of the HRG network.
		- node_positions: retuns 1D arrays with the coordinated X and Y of the nodes of tha spatial network models.
		- hyperbolic_distance: calculates the hyperbolic distane between two nodes.
		- erdos_renyi_network: returns an ER NetworkX graph .
		- random_geometric_graph: returns a RGG in NetworkX.
		- random_hyperbolic_graph: returns a HRG in NetworkX.
		- visualize_nonzero_elements: plots the nonzero elements of the adjacency matrix array.
		- visualize_network: returns a simple visualization of the chosen graph in its embedding space (if spatial) or fixed seed position if probabilistic.
		"""
		
		self.num_nodes = num_nodes
		self.model = model
		#self.graph = self.generate_network(**kwargs)
		self.kwargs = kwargs

	def generate_network(self):
		"""
		Generate the network based on the selected model.
		:return: NetworkX graph.
		"""
		if self.model == "ER":
			p = self.kwargs.get("p", 0.1)  # Default probability for ER model
			return self.erdos_renyi_network(p)
		elif self.model == "RGG":
			radius = self.kwargs.get("radius", 0.1)
			return self.random_geometric_graph(radius)
		#elif self.model == "dER":
		#    p = self.kwargs.get("p", 0.1)  # Default probability for ER model
		#    return self.directed_erdos_renyi_network(p)
		#elif self.model == "dRGG":
		#    p = self.kwargs.get("", 0.1)  # Default probability for ER model
		#    return self.random_geometric_graph()
		elif self.model == "HRG":
			curvature_param = self.kwargs.get("curvature_param", "N/A")
			disk_radius = self.kwargs.get("disk_radius", "N/A")
			alpha = self.kwargs.get("alpha", "N/A")
			return self.hyperbolic_random_graph(alpha, disk_radius, curvature_param)
		else:
			raise ValueError(f"Unsupported model: {self.model}")
			
	def adjacency_matrix_er(self):
		"""
		Create a binary and a weighted adjacency matrix of a ER graph with probability "p".
		:return: Two numpy arrays (dim, dim), dim = num_nodes. A weighted and a binary adjacency matrix.
		"""
		p = self.kwargs.get("p", "N/A")
		Weighted_Adjacency_matrix = np.zeros((self.num_nodes,self.num_nodes))
		Binary_Adjacency_matrix = np.zeros((self.num_nodes,self.num_nodes))
		for i in range(self.num_nodes):
			for j in range(i,self.num_nodes):
				ale = rd.random()
				if ale<=p: # Here, we define the connections (1 entries in the adjacency matrix) accorind to probability p
					Weighted_Adjacency_matrix[i,j] = np.random.normal(loc=0.0,scale=1.0) # Off-diagonal elements of the matrix
					#are weighted by a normal distribution with mean 0 and variance 1
					Weighted_Adjacency_matrix[j,i] = Weighted_Adjacency_matrix[i,j]
					Binary_Adjacency_matrix[i,j] = 1
					Binary_Adjacency_matrix[j,i] = Binary_Adjacency_matrix[i,j]
				
			Weighted_Adjacency_matrix[i,i] = np.random.normal(loc=0.0,scale=2.0) #Main diagonal elements are wighted with variance 2
			Binary_Adjacency_matrix[i,i] = 0
		return Weighted_Adjacency_matrix, Binary_Adjacency_matrix

	def adjacency_matrix_rgg(self):
		"""
		Create a binary and a weighted adjacency matrix of a RGG with connection radius "radius".
		:return: Two numpy arrays (dim, dim), dim = num_nodes. A weighted and a binary adjacency matrix.
		"""
		radius = self.kwargs.get("radius", "N/A")
		x, y = self.node_positions() 
		Weighted_Adjacency_matrix = np.zeros((self.num_nodes,self.num_nodes))
		Binary_Adjacency_matrix = np.zeros((self.num_nodes, self.num_nodes))
		for i in range(0, self.num_nodes):
			for j in range(i+1, self.num_nodes): 
				if np.sqrt(abs(  (x[i]-x[j])**2 + (y[i]-y[j])**2  )) < radius: #The distance between pairs of nodes should be equal less than radius
					Binary_Adjacency_matrix[i,j] = 1
					Binary_Adjacency_matrix[j,i] = Binary_Adjacency_matrix[i,j]
					Weighted_Adjacency_matrix[i,j] = np.random.normal(loc=0.0,scale=1.0) # Off-diagonal elements of the matrix
					Weighted_Adjacency_matrix[j,i] = Weighted_Adjacency_matrix[i,j]
			Binary_Adjacency_matrix[i,i] = 0
			Weighted_Adjacency_matrix[i,i] = np.random.normal(loc=0.0,scale=2.0) #Main diagonal elements are wighted with variance 2
		return Weighted_Adjacency_matrix, Binary_Adjacency_matrix
			
	def adjacency_matrix_hrg(self):
		"""
		Create a binary and a weighted adjacency matrix of a HRG with disk radius "disk_radius", curvature parameter "curvature_param", and alpha "alpha".
		:return: Two numpy arrays (dim, dim), dim = num_nodes. A weighted and a binary adjacency matrix.
		"""
		curvature_param = self.kwargs.get("curvature_param", "N/A")
		disk_radius = self.kwargs.get("disk_radius", "N/A")
		
		Weighted_Adjacency_matrix = np.zeros((self.num_nodes, self.num_nodes))
		Binary_Adjacency_matrix = np.zeros((self.num_nodes, self.num_nodes))
		
		for i in range(self.num_nodes):
			for j in range(i+1, self.num_nodes):
				distance = self.hyperbolic_distance(i, j, curvature_param)
				if  distance <= disk_radius:
					Binary_Adjacency_matrix[i,j] = 1
					Binary_Adjacency_matrix[j,i] = Binary_Adjacency_matrix[i,j]
					Weighted_Adjacency_matrix[i,j] = np.random.normal(loc=0.0,scale=1.0) # Off-diagonal elements of the matrix
					Weighted_Adjacency_matrix[j,i] = Weighted_Adjacency_matrix[i,j]
			   
			Weighted_Adjacency_matrix[i,i] = np.random.normal(loc=0.0, scale=2.0) #Main diagonal elements are wighted with variance 2
			Binary_Adjacency_matrix[i,i] = 0
		return Weighted_Adjacency_matrix, Binary_Adjacency_matrix
	   
	# Method to compute hyperbolic distance between two nodes
	def hyperbolic_distance(self, node_1, node_2, curvature_param):
		"""
		Calculate the hyperbolic ditance between two nodes.
		:param node_1: Label of the first node.
		:param node_2: Label of the second node.
		:param curvature_param: Parameter that governs the gaussian curvature of the embedding space as K=-(curvature_param**2).
		:return: A float with the hyperbolic distance.
		"""
		# Get positions of the nodes
		X, Y = self.node_positions()
		x_1, y_1 = X[node_1], Y[node_1]
		x_2, y_2 = X[node_2], Y[node_2]
		# Convert to polar coordinates
		r_1 = np.sqrt(x_1**2 + y_1**2)
		theta_1 = np.arctan2(y_1, x_1)
		r_2 = np.sqrt(x_2**2 + y_2**2)
		theta_2 = np.arctan2(y_2, x_2)
	
		# Hyperbolic distance formula
		deltaabs = np.pi - abs(np.pi - abs(theta_1 - theta_2)) #Angle difference between angular coordinates
		distance = (1 / curvature_param) * np.arccosh(
		np.cosh(curvature_param * r_1) * np.cosh(curvature_param * r_2) -
		np.sinh(curvature_param * r_1) * np.sinh(curvature_param * r_2) * np.cos(deltaabs)
		)
		return distance

	def node_positions(self):
		"""
		Calculate the random positions of the spatial random network models according to their distributions
		return: Two 1D numpy arrays with the X- and Y-coordinates of the nodes.
		"""
		x_coord = np.zeros((self.num_nodes))
		y_coord = np.zeros((self.num_nodes))
		if self.model == "RGG":
			x_coord = np.random.random(self.num_nodes)
			y_coord = np.random.random(self.num_nodes)
		elif self.model == "HRG":
			alpha = self.kwargs.get("alpha", "N/A")
			disk_radius = self.kwargs.get("disk_radius", "N/A")
			delta = np.cosh(alpha * disk_radius) - 1
			
			theta = np.zeros(self.num_nodes)
			radius = np.zeros(self.num_nodes)
			for i in range(self.num_nodes):
				theta[i] = rd.random() * 2 * np.pi  #Uniform distribution of angular coordinates in the hyperbolic disk
				radius[i] = (1 / alpha) * np.arccosh(delta * rd.random() + 1) #Exponential radii distribution in the hyperbolic disk
				x_coord[i] = radius[i] * np.cos(theta[i])
				y_coord[i] = radius[i] * np.sin(theta[i])
		return x_coord, y_coord
			
		
	def erdos_renyi_network(self,p):
		"""
		Generate an Erdős-Rényi random graph from its adjacency matrix.
		:param p: Probability of edge creation.
		:return: NetworkX graph.
		"""
		_ , binary_adj_matrix = self.adjacency_matrix_er()
		G = nx.from_numpy_array(binary_adj_matrix)
		return G

	def random_geometric_graph(self,radius):
		"""
		Generate a Random geometric graph from its adjacency matrix.
		:param radius: Connection radius.
		:return: NetworkX graph.
		"""
		_ , binary_adj_matrix = self.adjacency_matrix_rgg()
		G = nx.from_numpy_array(binary_adj_matrix)
		return G

	def hyperbolic_random_graph(self, alpha, disk_radius, curvature_param):
		"""
		Generate a hyperbolic random graph from its adjacency matrix.
		:param alpha: The exponent of the radii distribution coordinates of the nodes inside the hyperbolic disk.
		:param disk_radius: Radius of the hyperbolic disk where nodes are embedded, it is also the connection radius.
		:param curvature_param: Parameter that governs the gaussian curvature of the embedding space as K=-(curvature_param**2) .
		:return: NetworkX graph.
		"""
		_ , binary_adj_matrix = self.adjacency_matrix_hrg()
		G = nx.from_numpy_array(binary_adj_matrix)
		return G
			
					
	def visualize_network(self, G, ax=None, node_size: int = 100):
		"""
		Visualize the network.
		:param G: NetworkX graph to visualize.
		return: Plot of the graph.
		"""
		
		if ax is None:
			# Create a new figure if no axis is provided
			fig, ax = plt.subplots(figsize=(6, 6))
		if self.model == "ER":
			pos = nx.spring_layout(nx.complete_graph(self.num_nodes), seed=48)
		elif self.model == "RGG":
			x, y = self.node_positions()
			pos = {i: (x[i], y[i]) for i in range(len(x))}
		elif self.model == "HRG":
			x, y = self.node_positions()
			pos = {i: (x[i], y[i]) for i in range(len(x))}
		else:
			raise ValueError(f"Unsupported model: {self.model}")
		p = self.kwargs.get("p", "N/A")
		radius = self.kwargs.get("radius", "N/A")
			
		if self.model == "ER":
			ax.set_title(f"{self.model} Network ($n$={self.num_nodes} and $p$={p})")
			nx.draw(G, pos, ax=ax, with_labels=False, node_color='lightblue', edge_color='gray', node_size = node_size)
		elif self.model == "RGG":
			ax.set_title(f"{self.model} ($n$={self.num_nodes} and $\\ell$={radius})")
			# Draw the enclosing square from (0,0) to (1,1)
			ax.set_xlim(0, 1)
			ax.set_ylim(0, 1)
			#plt.gca().set_aspect('equal', adjustable='box')
			ax.add_patch( (plt.Rectangle((0, 0), 1, 1, edgecolor='black', fill=False, linewidth=1)) )
			nx.draw_networkx_nodes(G,pos,ax=ax, node_color='skyblue', node_size=node_size)
			nx.draw_networkx_edges(G, pos, edge_color='gray', width=0.5, ax=ax)
		elif self.model == "HRG":
			curvature_param = self.kwargs.get("curvature_param", "N/A")
			disk_radius = self.kwargs.get("disk_radius", "N/A")
			alpha = self.kwargs.get("alpha", "N/A")
			ax.set_title(f"{self.model} ($n$={self.num_nodes}, $\\alpha$={alpha}, $K$={-curvature_param**2}, and $\\rho$={disk_radius} )")
			nx.draw(G, pos, ax = ax, with_labels=False, node_color='black', edge_color='c', node_size=node_size, width=0.25)
			# Draw the reference circle
			circle = plt.Circle((0, 0), disk_radius, color='red', fill=False, linestyle='--')
			ax.add_artist(circle)
		return ax
		
		
	def visualize_nonzero_elements(self, adj_matrix, ax=None, cmap_name: str = "Blues"):
		if ax is None:
			fig, ax = plt.subplots(figsize=(6,6))
		ax.imshow(adj_matrix, cmap=cmap_name)
		return ax
			

		
			
class StructuralPropertiesRandomNetworks(RandomNetworkModels):
	"""
	A subclass of RandomNetworkModels for the calculation of structural properties of networks such as: 
	-Randic and Harmonic topological indices
	-Average degree
	-Number of nonisolated vertices
	-Global clustering coefficient
	"""
	def __init__(self, num_nodes: int, model: str = "ER", **kwargs):
		"""
		Initialize the network with a given model.

		Parameters:
		- num_nodes: The total number of nodes in the network.
		- model: Type of random network ('ER' for Erdős-Rényi,
				 'RGG' for directed Random Geometric Graph,
				 "HRG" for hyperbolic random graph).
		- kwargs: Additional parameters for network generation.

		Methods:
		- degrees_of_the_network: retuns a 1D NumPy array containing the degrees of all nodes.
		- average_degree: returns a float contining the average degree.
		- topological_indices: returns two floats containing the values of the Randic and Harmonic index.
		- number_of_nonisolated_vertices: returns a float containing the number of nonisolated vertices.
		- clustering_coeffients: returns two floats contining the average local clustering coefficient and the global clustering coefficient.
		"""
		super().__init__(num_nodes, model, **kwargs)  # We call the parent class's __init__ method
			
	
	def degrees_of_the_network(self, binary_adj_matrix):
		"""
		Calculate the degrees of all nodes from the binary adjacency matrix.
		:param binary_adj_matrix: Binary adjacency matrix of the network.
		:return: A 1D NumPy array containing the degrees of all nodes.
		"""
		degrees = np.sum(binary_adj_matrix, axis=1)
		return degrees
			
	def average_degree(self, degrees):
		"""
		Calculate the average degree of the network from the binary adjacency matrix.
		:param binary_adj_matrix: Binary adjacency matrix of the network.
		:return: A float contining the average degree.
		"""
		average_degree = np.sum(degrees)/(self.num_nodes)
		return average_degree
			
	#We compute the Randic index and Harmonic index directly from the binary adjacency matrix of the model
	def topological_indices(self, binary_adj_matrix):
		"""
		Calculate the Randic and Harmonic indices from the binary adjacency matrix.
		:param binary_adj_matrix: Binary adjacency matrix of the network.
		:return: Two floats contining the values of the Randic and Harmonic index (in that order).
		"""
		Randic_index = 0
		Harmonic_index = 0
		degrees = self.degrees_of_the_network(binary_adj_matrix)
		for i in range(self.num_nodes):
			Randic = 0
			Harmonic = 0
			for j in range(i, self.num_nodes):
				if binary_adj_matrix[i,j] != 0:
					Randic += 1.0/( np.sqrt(degrees[i]*degrees[j]) )
					Harmonic += 2.0/( degrees[i] + degrees[j] )
			Randic_index += Randic
			Harmonic_index += Harmonic
		return Randic_index, Harmonic_index

	def number_of_nonisolated_vertices(self, binary_adj_matrix):
		"""
		Calculate the number of nonisolated vertices of the network from the binary adjacency matrix.
		:param binary_adj_matrix: Binary adjacency matrix of the network.
		:return: A float contining the number of nonisolated vertices.
		"""
		degrees = self.degrees_of_the_network(binary_adj_matrix)
		nonisolated_nodes_counter = 0
		for i in range(self.num_nodes):
			if degrees[i] != 0:
				nonisolated_nodes_counter += 1
		return nonisolated_nodes_counter

	#With this method we can compute the local and global clustering coeffients from the adjacency matrix
	def clustering_coeffients(self, binary_adj_matrix):
		"""
		Calculate the average local and global clustering coefficients of the network from the binary adjacency matrix.
		:param binary_adj_matrix: Binary adjacency matrix of the network.
		:return: Two floats contining the average local clustering coefficient and the global clustering coefficient (in that order).
		"""
		degrees = self.degrees_of_the_network(binary_adj_matrix)
		local_clustering_node = np.zeros((self.num_nodes))
		product_of_degrees = np.zeros((self.num_nodes))
		product_of_degrees[:] = degrees[:]*(degrees[:]-1)
		sum_product_of_degrees = np.sum(product_of_degrees)
		product_G = 0
		for i in range(self.num_nodes):
			product_k = 0
			for j in range(self.num_nodes):
				for k in range(self.num_nodes):
					 product_k += binary_adj_matrix[i,j]*binary_adj_matrix[j,k]*binary_adj_matrix[k,i]
					 product_G += binary_adj_matrix[i,j]*binary_adj_matrix[j,k]*binary_adj_matrix[k,i]
			if  (degrees[i] == 0) or (degrees[i] == 1) :
				local_clustering_node[i] = 0
			else:
				local_clustering_node[i] = (product_k) / ( product_of_degrees[i] )

		average_local_clustering_coefficient = np.sum(local_clustering_node)/(self.num_nodes)
			
			
		if np.sum(product_of_degrees) == 0:
			global_clustering_coefficient = 0
		else:
			global_clustering_coefficient = product_G / sum_product_of_degrees
		return average_local_clustering_coefficient, global_clustering_coefficient
		   

			
			
			
class SpectralPropertiesRandomNetworks(RandomNetworkModels):
	"""
	A subclass of RandomNetworkModels for the calculation of spectral properties of networks such as: 
	-Shannon entropy
	-Participation ratios
	-Ratio between consecutive eigenvalue spacings
	"""
	def __init__(self, num_nodes: int, model: str = "ER", **kwargs):
		"""
		Initialize the network with a given model.

		Parameters:
		- num_nodes: The total number of nodes in the network.
		- model: Type of random network ('ER' for Erdős-Rényi,
				 'RGG' for directed Random Geometric Graph,
				 "HRG" for hyperbolic random graph).
		- kwargs: Additional parameters for network generation.

		Methods:
		- diagonalize_weighted_adj_matrix: returns a 1D array with the eigenvalues of the matrix, a (dim, dim) matrix with the normalized eigenvectors
		in the columns.
		- avoid_numerical_precision_errors: returns refinement of the values of each eigenvector element to avoid precision errors.
		- shannon_entropy_participation_ratio: returns two floats containing the values of the average Shannon entropy and participation ratio of the network.
		- ratio_between_consecutive_eigenvalue_spacings: returns a float containing the average value of the ratio.
		"""
		super().__init__(num_nodes, model, **kwargs)  # We call the parent class's __init__ method
	def diagonalize_weighted_adj_matrix(self, weighted_adj_matrix):
		# Eigenvalue decomposition
		EigV, U = eigh(weighted_adj_matrix) #EigV stands for eigenvalues and U is the matrix that contains the eigenvectors as columns.
		return EigV, U
			
	def avoid_numerical_precision_errors(self,U):
		threshold = 1.0e-30  # Threshold for considering a value as "too small"
		replacement_value = 1e-14  # Replacement value for very small entries
		U[np.abs(U)**2 <= threshold] = replacement_value
		return U
			
			
	def shannon_entropy_participation_ratio(self, U):
		U = self.avoid_numerical_precision_errors(U)
		shannon_entropy = 0
		participation_ratio = 0
		for i in range(self.num_nodes):
			shannon_entropy -=   np.sum( (np.abs(U[:,i])**2) * np.log(np.abs(U[:,i])**2)  )
			participation_ratio += 1.0/( np.sum( np.abs(U[:,i])**4 ) )
		return shannon_entropy/self.num_nodes, participation_ratio/self.num_nodes

	def ratio_between_consecutive_eigenvalue_spacings(self,EigV):
		ratio = 0
		for i in range(1, self.num_nodes-1):
			top_difference = EigV[i+1]-EigV[i]
			bottom_difference = EigV[i]-EigV[i-1]
				
				
			if top_difference <= bottom_difference:
				smallest_difference = top_difference
				largest_difference = bottom_difference
			else:
				smallest_difference = bottom_difference
				largest_difference = top_difference
			ratio += smallest_difference/largest_difference
		return ratio/(self.num_nodes -2)
			
			
			
			


