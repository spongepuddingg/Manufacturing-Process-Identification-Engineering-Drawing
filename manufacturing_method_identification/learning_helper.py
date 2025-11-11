"""Learning Helper Functions

Utility functions for training and data processing in the manufacturing
method identification pipeline.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_sequence
from torchvision import transforms
import os
import numpy as np
from clusterData import *
from model import *
import argparse
import time
import sklearn.metrics as metrics
import PIL
from math import *
import numpy as np
import math
import json

def get_angle(x1, x2, y1, y2, x3, x4, y3, y4):
	"""Calculate angle between two line segments using slopes
	
	Args:
		x1, y1, x2, y2: First line segment coordinates
		x3, y3, x4, y4: Second line segment coordinates
		
	Returns:
		Angle between line segments in radians
	"""
	m1 = (y2 - y1) / (x2 - x1)
	m2 = (y4 - y3) / (x4 - x3)
	k = (m1 - m2) / (1 + m1 * m2)
	if k < 0:
		k = -1 * k
	a = np.arctan(k)
	return a

def my_collate(batch):
	"""Custom collate function for batching graph data (unused in current implementation)
	
	Args:
		batch: List of data samples
		
	Returns:
		tuple: Packed data, batch indices, edge indices, targets
	"""
	data = [item[0] for item in batch]
	data_length = [len(sq) for sq in data]
	cnt = 0
	gcn_batch = []
	
	# Create batch indices for graph neural network
	for length in data_length:
		y = [cnt] * length
		gcn_batch.extend(y)
		cnt += 1
	
	gcn_batch = torch.tensor(gcn_batch, dtype=torch.long)
	data = pack_sequence(data, enforce_sorted=False)
	targets = [item[1] for item in batch]
	targets = pack_sequence(targets, enforce_sorted=False).data 
	targets = targets.long()

	# Calculate edge attributes using k-NN graph
	knn = DilatedKnnGraph(args.k, 1, True, 0.3)
	edge_index = knn(data.data, gcn_batch)
	return data, gcn_batch, edge_index, targets


def cluster_collate(batch):
	"""Custom collate function for clustering data with edge attributes (unused)
	
	Args:
		batch: List of data samples with original data
		
	Returns:
		list: Processed data with edge attributes and original data
	"""
	data = [item[0] for item in batch]
	data_length = [len(sq) for sq in data]
	cnt = 0
	gcn_batch = []
	
	# Create batch indices
	for length in data_length:
		y = [cnt] * length
		gcn_batch.extend(y)
		cnt += 1
	
	gcn_batch = torch.tensor(gcn_batch, dtype=torch.long)
	data = pack_sequence(data, enforce_sorted=False)
	targets = [item[1] for item in batch]
	targets = pack_sequence(targets, enforce_sorted=False).data 
	targets = targets.long()

	# Pack original data
	data_orin = [item[2] for item in batch]
	data_orin = pack_sequence(data_orin, enforce_sorted=False)

	# Calculate edge attributes with angles
	knn = DilatedKnnGraph(3, 1, True, 0.3)
	edge_index = knn(data.data, gcn_batch)
	edge_attr = []
	
	for i in range(edge_index.shape[1]):
		idx1, idx2 = edge_index[0, i].item(), edge_index[1, i].item()
		x1, x2, y1, y2 = data.data[idx1, :] 
		x3, x4, y3, y4 = data.data[idx2, :]
		angle = get_angle(x1, x2, y1, y2, x3, x4, y3, y4)
		if math.isnan(angle):
			angle = 0
		edge_attr.append(angle)
	
	edge_attr = torch.tensor(edge_attr, dtype=torch.float)
	edge_attr = edge_attr.view(-1, 1)
	return [data, gcn_batch, edge_index, edge_attr, targets, data_orin]


def create_model(args):
	"""Create and initialize DeepGCN model
	
	Args:
		args: Command line arguments with model configuration
		
	Returns:
		Initialized model on specified device
	"""
	model = DeepGCN(args).to(device)
	print(f'Total parameters: {sum([p.numel() for p in model.parameters()]):,}')
	return model 

def create_optimizer(args, model):
	"""Create optimizer based on command line arguments
	
	Args:
		args: Command line arguments with optimizer configuration
		model: Model to optimize
		
	Returns:
		Configured optimizer
	"""
	if args.optim == 'sgd':
		return optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd)
	elif args.optim == 'adam':
		return optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
	elif args.optim == "adadelta":
		return optim.Adadelta(model.parameters())


