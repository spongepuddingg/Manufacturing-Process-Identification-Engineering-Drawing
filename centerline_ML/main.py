"""Centerline Detection Training Script

This script trains a GCN model to classify line segments in technical drawings
as centerlines. The model uses graph neural networks to process spatial relationships
between line segments.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch_geometric.transforms as T
from torch.optim.lr_scheduler import MultiStepLR
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_sequence
from torchvision import transforms
from torch_geometric.data import Data, DataLoader
from torch_cluster import knn_graph
import os
import numpy as np
from model import *
import argparse
import time
import sklearn.metrics as metrics
import PIL
from gcn_lib.sparse import MultiSeq, MLP, GraphConv, PlainDynBlock, ResDynBlock, DenseDynBlock, DilatedKnnGraph
from math import *
import numpy as np
import math
import json
# Command line argument parser for training configuration
parser = argparse.ArgumentParser(description='Centerline Detection Training')
parser.add_argument('--wd', default=2e-4, type=float, help='Weight decay')
parser.add_argument('--lr', default=1e-3, type=float, help='Initial learning rate')
parser.add_argument('--lr_decay', default=0.7, type=float, help='Multiplicative factor used on learning rate at `lr_steps`')
parser.add_argument('--lr_steps', default='[100, 150, 175]', help='List of epochs where the learning rate is decreased by `lr_decay`')
parser.add_argument('--momentum', default=0.9, type=float, help='Momentum')
parser.add_argument('--epochs', default=10, type=int, help='Number of epochs to train. If <=0, only testing will be done.')
parser.add_argument('--optim', default='adam', help='Optimizer: sgd|adam|adadelta')
parser.add_argument('--grad_clip', default=1, type=float, help='Element-wise clipping of gradient. If 0, does not clip')
parser.add_argument('--loss_weights', default='none', help='[none, proportional, sqrt] how to weight the loss function')

# dataset arguments
parser.add_argument('--batch_size', default=16, type=int, help='mini-batch size (default:16)')
parser.add_argument('--in_channels', default=4, type=int, help='the channel size of input point cloud ')
parser.add_argument('--n_classes', default=3, type=int, help='number of classes for the line clusters')

# model args
parser.add_argument('--pretrained_model', type=str, help='path to pretrained model(default: none)', default='')
parser.add_argument('--k', default=16, type=int, help='neighbor num (default:16)')
parser.add_argument('--block', default='plain', type=str, help='graph backbone block type {plain, res, dense}')
parser.add_argument('--conv', default='edge', type=str, help='graph conv layer {edge, mr}')
parser.add_argument('--act', default='leakyrelu', type=str, help='activation layer {relu, prelu, leakyrelu}')
parser.add_argument('--norm', default='batch', type=str, help='{batch, instance} normalization')
parser.add_argument('--bias', default=True,  type=bool, help='bias of conv layer True or False')
parser.add_argument('--n_filters', default=32, type=int, help='number of channels of deep features')
parser.add_argument('--n_blocks', default=6, type=int, help='number of basic blocks')
parser.add_argument('--dropout', default=0.3, type=float, help='ratio of dropout')

# dilated knn
parser.add_argument('--epsilon', default=0.2, type=float, help='stochastic epsilon for gcn')
parser.add_argument('--stochastic', default=True,  type=bool, help='stochastic for gcn, True or False')
parser.add_argument('--use_dilation', default=True,  type=bool, help='dilation')

args = parser.parse_args()


# Setup device and training parameters
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

# Training configuration
params = {'batch_size': args.batch_size,
			'shuffle': True,
			'num_workers': 8}
max_epochs = 200

# Loss function for multi-class classification
criterion = torch.nn.CrossEntropyLoss().to(device)

# Load training data from numpy files
# Currently only loading Lathe data for centerline detection
from glob import glob
data_vars = {}
for filename in glob('data/Lathe/*.np[yz]'):
	data_vars[filename] = np.load(filename)

# Split data into train/test sets
train_data = np.array(list(data_vars.values()))
test_data = train_data[:30]  # First 30 samples for testing
train_data = train_data[30:]  # Remaining samples for training

# Create PyTorch Geometric data objects with graph structure
train_list, test_list = [], []

# Data augmentation for training: random translation + normalization
transform_train = T.Compose([
	T.RandomTranslate(0.1), 
	T.NormalizeScale(), 
	T.TargetIndegree(norm=True, max_value=None, cat=True)
])

# Test data transformation: only normalization
transform_test = T.Compose([
	T.NormalizeScale(), 
	T.TargetIndegree(norm=True, max_value=None, cat=True)
])

# Graph construction using k-nearest neighbors
pre_transform = T.KNNGraph(k=6)

# Process training data
for data in train_data:
	# Create graph data: x=coordinates, y=labels
	train_data_item = Data(
		x=torch.tensor(data[:, :4], dtype=torch.float),  # Line coordinates (x1,y1,x2,y2)
		y=torch.tensor(data[:, 4], dtype=torch.float)    # Centerline labels
	)
	train_data_item.pos = train_data_item.x  # Position for graph construction
	train_data_item = pre_transform(train_data_item)  # Build k-NN graph
	train_data_item = transform_train(train_data_item)  # Apply augmentation
	train_list.append(train_data_item)

# Process test data
for data in test_data:
	test_data_item = Data(
		x=torch.tensor(data[:, :4], dtype=torch.float),
		y=torch.tensor(data[:, 4], dtype=torch.float)
	)
	test_data_item.pos = test_data_item.x
	test_data_item = pre_transform(test_data_item)
	test_data_item = transform_test(test_data_item)
	test_list.append(test_data_item)



# Create data loaders for batch processing
training_generator = DataLoader(train_list, batch_size=16, shuffle=True)
testing_generator = DataLoader(test_list, batch_size=16, shuffle=True)

print(f"Training samples: {len(train_list)}")
print(f"Testing samples: {len(test_list)}")

# Model initialization and optimizer setup
def create_model(args):
	"""Initialize DeepGCN model with specified architecture"""
	model = DeepGCN(args).to(device)
	print(f'Total parameters: {sum([p.numel() for p in model.parameters()]):,}')
	return model 

def create_optimizer(args, model):
	"""Create optimizer based on command line arguments"""
	if args.optim == 'sgd':
		return optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd)
	elif args.optim == 'adam':
		return optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
	elif args.optim == "adadelta":
		return optim.Adadelta(model.parameters())

model = create_model(args)
optimizer = create_optimizer(args, model)

# Training loop
best_testacc = 0
for epoch in range(max_epochs):
	train_pred = []
	train_true = []
	model.train()
	train_loss = 0.0
	count = 0.0
	batch_num = 0
	for data in training_generator:
		# Move data to GPU
		data = data.to(device)
		batch_size = 16
		
		# Skip incomplete batches
		if torch.max(data.batch) < 15:
			continue
		
		# Forward pass
		optimizer.zero_grad()
		outputs = model(data.pos, data.batch, data.edge_index, data.edge_attr)
		loss = criterion(outputs, data.y.long())
		print(f'Batch loss: {loss.item():.6f}')
		
		# Backward pass
		loss.backward()
		optimizer.step()

		preds = outputs.max(dim=1)[1]
		count += batch_size
		train_loss += loss.item() * batch_size
		train_true.append(data.y.long().cpu().numpy())
		train_pred.append(preds.detach().cpu().numpy())
		batch_num+=1
	# Calculate training metrics
	train_true = np.concatenate(train_true)
	train_pred = np.concatenate(train_pred)
	
	train_acc = metrics.accuracy_score(train_true, train_pred)
	train_balanced_acc = metrics.balanced_accuracy_score(train_true, train_pred)
	
	print(f'Epoch {epoch}: Loss={train_loss/count:.6f}, Acc={train_acc:.6f}, Balanced Acc={train_balanced_acc:.6f}')
		
	if epoch % 1 == 0:
		with torch.set_grad_enabled(False):
			test_pred = []
			test_true = []
			model.eval()
			#test_loss = 0.0
			#count = 0.0

		for data in testing_generator:
			# Transfer to GPU
			data = data.to(device)
			if torch.max(data.batch) < 15:
				continue
			# Forward pass for evaluation
			outputs = model(data.pos, data.batch, data.edge_index, data.edge_attr)
			test_preds = outputs.max(dim=1)[1]
			test_true.append(data.y.long().cpu().numpy())
			test_pred.append(test_preds.detach().cpu().numpy())
		# Calculate test metrics
		test_true = np.concatenate(test_true)
		test_pred = np.concatenate(test_pred)
		
		test_acc = metrics.accuracy_score(test_true, test_pred)
		test_balanced_acc = metrics.balanced_accuracy_score(test_true, test_pred)
		
		print(f'Test Epoch {epoch}: Acc={test_acc:.6f}, Balanced Acc={test_balanced_acc:.6f}')
		
		# Save best model based on combined train+test balanced accuracy
		combined_score = test_balanced_acc + train_balanced_acc
		if combined_score >= best_testacc:
			best_testacc = combined_score
			torch.save(model.state_dict(), "saved_model/best_test.pt")
			print("Model saved!")

print(f"Best combined balanced accuracy: {best_testacc:.6f}")

		