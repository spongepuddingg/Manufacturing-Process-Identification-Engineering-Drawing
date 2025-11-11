"""Manufacturing Method Classifier

Implements a graph neural network classifier for identifying manufacturing
methods from technical drawings using SAGE convolution and differentiable pooling.
"""

import torch.nn as nn
import torch
import torch.nn.functional as F
from torch_geometric.nn import DenseSAGEConv, dense_diff_pool
from torch.autograd import Variable
import torch_geometric.utils as utils

device = torch.device('cuda:0')

class GNN(torch.nn.Module):
	"""Graph Neural Network module using SAGE convolution layers
	
	Args:
		in_channels: Number of input features
		hidden_channels: Number of hidden features
		out_channels: Number of output features
		normalize: Whether to normalize features
		add_loop: Whether to add self-loops
		lin: Whether to include linear transformation
	"""
	
	def __init__(self, in_channels, hidden_channels, out_channels, 
				 normalize=False, add_loop=False, lin=True):
		super(GNN, self).__init__()

		self.add_loop = add_loop

		# Three SAGE convolution layers with batch normalization
		self.conv1 = DenseSAGEConv(in_channels, hidden_channels, normalize)
		self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
		self.conv2 = DenseSAGEConv(hidden_channels, hidden_channels, normalize)
		self.bn2 = torch.nn.BatchNorm1d(hidden_channels)
		self.conv3 = DenseSAGEConv(hidden_channels, out_channels, normalize)
		self.bn3 = torch.nn.BatchNorm1d(out_channels)

			# Optional linear layer for feature combination
		if lin is True:
			self.lin = torch.nn.Linear(2 * hidden_channels + out_channels, out_channels)
		else:
			self.lin = None
	def bn(self, i, x):
		"""Apply batch normalization to dense tensor"""
		batch_size, num_nodes, num_channels = x.size()
		
		# Reshape for batch normalization
		x = x.view(-1, num_channels)
		x = getattr(self, f'bn{i}')(x)
		x = x.view(batch_size, num_nodes, num_channels)
		return x
	def forward(self, x, adj, mask=None):
		"""Forward pass through GNN layers
		
		Args:
			x: Node features
			adj: Adjacency matrix
			mask: Optional mask for nodes
			
		Returns:
			Processed node features
		"""
		x0 = x
		
		# Three convolution layers with skip connections
		x1 = self.bn(1, F.leaky_relu(self.conv1(x0, adj, mask)))
		x2 = self.bn(2, F.leaky_relu(self.conv2(x1, adj, mask)))
		x3 = self.bn(3, F.leaky_relu(self.conv3(x2, adj, mask)))
		
		# Concatenate features from all layers
		x = torch.cat([x1, x2, x3], dim=-1)
		
		# Optional linear transformation
		if self.lin is not None:
			x = F.leaky_relu(self.lin(x))
		return x

class Classifier(torch.nn.Module):
	"""Manufacturing Method Classifier using hierarchical graph pooling
	
	Args:
		in_channels: Input feature dimensions
		hidden_channels: Hidden layer dimensions
		out_channels: Output feature dimensions
		feat_dims: Feature dimensions (unused in current implementation)
		num_class: Number of manufacturing classes (3: Lathe, Sheet Metal, Milling)
	"""
	
	def __init__(self, in_channels, hidden_channels, out_channels, feat_dims, num_class):
		super(Classifier, self).__init__()

		num_clusters = 5  # Number of clusters for differentiable pooling
		
		# GNN for generating cluster assignments
		self.gnn1_pool = GNN(in_channels, hidden_channels, num_clusters)
		
		# GNN for node embeddings
		self.gnn1_embed = GNN(in_channels, hidden_channels, hidden_channels, lin=False)
		
		# GNN for processing pooled features
		self.gnn3_embed = GNN(3 * hidden_channels, hidden_channels, out_channels, lin=False)
		
		# Classification layers
		self.lin1 = torch.nn.Linear(2 * hidden_channels + out_channels, out_channels)
		self.lin2 = torch.nn.Linear(out_channels, num_class)

	def forward(self, data):
		"""Forward pass through the classifier
		
		Args:
			data: PyTorch Geometric data object containing graph information
			
		Returns:
			tuple: (log_softmax_output, features, link_loss, entropy_loss)
		"""
		x, edge_index = data.x, data.edge_index
		
		# Convert to dense format for differentiable pooling
		dense_x = utils.to_dense_batch(x, batch=data.batch)
		x = dense_x[0]
		adj = utils.to_dense_adj(data.edge_index, batch=data.batch)
		
		# Generate cluster assignments and node embeddings
		s = self.gnn1_pool(x, adj)  # Cluster assignment matrix
		x = self.gnn1_embed(x, adj)  # Node embeddings
		
		# Apply differentiable pooling
		x, adj, l1, e1 = dense_diff_pool(x, adj, s)
		
		# Process pooled features
		x = self.gnn3_embed(x, adj)
		
		# Global pooling and classification
		x = x.mean(dim=1)  # Average pooling
		x1 = self.lin1(x)
		x = F.leaky_relu(x1)
		x = self.lin2(x)

		return F.log_softmax(x, dim=-1), x1, l1, e1


		