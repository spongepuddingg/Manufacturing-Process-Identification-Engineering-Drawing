"""Data Loading and Preprocessing for Manufacturing Method Classification

This module handles loading and preprocessing of manufacturing drawing data,
converting line segments into graph representations with geometric features.
"""

import numpy as np 
import torch
import json
import svgwrite
import torch.utils.data as data
from torch_geometric.data import DataLoader
from torch_geometric.data import Data
import torch
from gcn_lib.sparse import MultiSeq, MLP, GraphConv, PlainDynBlock, ResDynBlock, DenseDynBlock, DilatedKnnGraph
import math
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import pandas as pd 
import os
import random
from random import sample

class sketchData(data.Dataset):
	"""Dataset class for manufacturing drawing data
	
	Processes line segment data into graph representations with geometric features
	and constructs k-NN graphs based on spatial relationships.
	
	Args:
		data: Array of line segment coordinates
		names: Filenames corresponding to each data sample
		labels: Manufacturing method labels
		transform: Optional data transformation
		train: Whether this is training data
	"""
	
	def __init__(self, data, names, labels, transform=None, train=False):
		self.transform = transform
		self.train = train
		self.data = data
		self.labels = labels
		self.names = names
		
		if transform is not None:
			self.data = torch.from_numpy(np.array(list(map(transform, self.data)))).to(device)

	def __len__(self):
		return self.data.shape[0]

	def __getitem__(self, index):
		"""Get a single data sample with graph structure
		
		Args:
			index: Sample index
			
		Returns:
			PyTorch Geometric Data object with graph structure
		"""
		# Extract filename and load corresponding data
		name = self.names[index]
		data_item = torch.from_numpy(self.data[index])
		
		# Calculate scaling factors for normalization
		x1_min, y1_min, x2_min, y2_min = self.scaling_factor(data_item)
		feature = self.scale_stroke(data_item, x1_min, y1_min, x2_min, y2_min)
		
		# Remove any NaN values
		feature = feature[~torch.any(torch.isnan(feature), dim=1)]
		
		# Extract label from filename
		labels = self.labels
		name = os.path.splitext(name)[0]
		if '/' in name: 
			name = name.split('/')[-1]
		name = name + '.svg'
		labels = labels[labels['title'].str.contains(name)].label[0]
		
		# Limit number of nodes to prevent memory issues
		data_length = feature.shape[0]
		if data_length > 800:
			indice = random.sample(range(data_length), 800)
			indice = torch.tensor(indice)
			feature = feature[indice]
		
		# Construct k-NN graph
		knn = DilatedKnnGraph(16, 1, True, 0.3)
		edge_idx = knn(feature)
		
		# Calculate edge attributes (angles between connected line segments)
		edge_attr = []
		for i in range(edge_idx.shape[1]):
			idx1, idx2 = edge_idx[0, i].item(), edge_idx[1, i].item()
			x1, y1, x2, y2 = feature[idx1, :4] 
			x3, y3, x4, y4 = feature[idx2, :4]
			angle = self.get_angle(x1, x2, y1, y2, x3, x4, y3, y4)
			if math.isnan(angle):
				angle = 0
			edge_attr.append(angle)
		
		edge_attr = torch.tensor(edge_attr, dtype=torch.float)
		edge_attr = edge_attr.view(-1, 1)
		
		# Create PyTorch Geometric data object
		data = Data(x=feature, edge_index=edge_idx, edge_attr=edge_attr, y=labels)
		del feature, edge_idx, edge_attr, labels
		return data

	def scaling_factor(self, feature):
		"""Calculate scaling factors for coordinate normalization"""
		x1_max = torch.max(feature[:, 0])
		y1_max = torch.max(feature[:, 1])
		x2_max = torch.max(feature[:, 2])
		y2_max = torch.max(feature[:, 3])
		return [x1_max, y1_max, x2_max, y2_max]

	def scale_stroke(self, x, x1_max, y1_max, x2_max, y2_max):
		"""Normalize line segment coordinates to [0,1] range"""
		x = x.float()
		
		# Translate to origin
		x[:, 0] -= torch.min(x[:, 0])  # x1 coordinates
		x[:, 1] -= torch.min(x[:, 1])  # y1 coordinates
		x[:, 2] -= torch.min(x[:, 2])  # x2 coordinates
		x[:, 3] -= torch.min(x[:, 3])  # y2 coordinates

		# Scale to [0,1] range
		x[:, 0] = x[:, 0] / x1_max if x1_max != 0 else x[:, 0]
		x[:, 1] = x[:, 1] / y1_max if y1_max != 0 else x[:, 1]
		x[:, 2] = x[:, 2] / x2_max if x2_max != 0 else x[:, 2]
		x[:, 3] = x[:, 3] / y2_max if y2_max != 0 else x[:, 3]
		return x

	def get_angle(self, x1, x2, y1, y2, x3, x4, y3, y4):
		"""Calculate angle between two line segments
		
		Args:
			x1, y1, x2, y2: Coordinates of first line segment
			x3, y3, x4, y4: Coordinates of second line segment
			
		Returns:
			Angle between the two line segments in radians
		"""
		# Calculate slopes of both line segments
		m1 = (y2 - y1) / (x2 - x1)
		m2 = (y4 - y3) / (x4 - x3)
		
		# Calculate angle using slope difference formula
		k = (m1 - m2) / (1 + m1 * m2)
		if k < 0:
			k = -1.0 * k  # Take absolute value
		a = torch.atan(k)
		return a

	def get_orientation(self, x1, x2, y1, y2):
		"""Calculate orientation angle of a line segment
		
		Args:
			x1, y1, x2, y2: Line segment coordinates
			
		Returns:
			Orientation angle in degrees
		"""
		orientation = math.atan2(abs((y1 - y2)), abs((x1 - x2)))
		return math.degrees(orientation)