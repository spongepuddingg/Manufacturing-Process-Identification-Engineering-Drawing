"""Centerline Detection Inference Script

Runs inference on trained GCN model to classify line segments
as centerlines in technical drawings.
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
# Command line arguments for inference
parser = argparse.ArgumentParser(description='Centerline Detection Inference')
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
parser.add_argument('--pretrained_model', type=str, help='path to pretrained model(default: none)', default='/media/louise/ubuntu_work2/lost+found/centerline_ML/saved_model/best_test.pt')
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


# Setup device for inference
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

# Inference parameters
params = {'batch_size': args.batch_size,
			'shuffle': True,
			'num_workers': 8}
max_epochs = 200

# Loss function (for evaluation if ground truth available)
criterion = torch.nn.CrossEntropyLoss().to(device)

# Load test data for inference
from glob import glob
data_vars = {}
for filename in glob('data/Lathe/*.np[yz]'):
	data_vars[filename] = np.load(filename)

# Use first 32 samples for inference
data = np.array(list(data_vars.values()))
test_data = data[:32]

# Prepare test data for inference
train_list, test_list = [], []

# Data transformations for inference
transform_test = T.Compose([
	T.NormalizeScale(), 
	T.TargetIndegree(norm=True, max_value=None, cat=True)
])
pre_transform = T.KNNGraph(k=6)

# Process each test sample
for data in test_data:
	# Handle data with or without ground truth labels
	if data.shape[1] == 5:  # Has labels
		test_data_item = Data(
			x=torch.tensor(data[:, :4], dtype=torch.float), 
			y=torch.tensor(data[:, 4], dtype=torch.float)
		)
	else:  # No labels available
		test_data_item = Data(
			x=torch.tensor(data[:, :4], dtype=torch.float), 
			y=torch.zeros(len(data))
		)
	
	# Set position for graph construction
	test_data_item.pos = test_data_item.x
	test_data_item = pre_transform(test_data_item)
	test_data_item = transform_test(test_data_item)
	test_list.append(test_data_item)
# Create data loader for inference
testing_generator = DataLoader(test_list, batch_size=args.batch_size, shuffle=False)
print(f"Number of test samples: {len(test_list)}")

# Model initialization for inference
def create_model(args):
	"""Create and load pretrained model"""
	model = DeepGCN(args).to(device)
	print(f'Total parameters: {sum([p.numel() for p in model.parameters()]):,}')
	
	# Load pretrained weights
	if args.pretrained_model:
		model.load_state_dict(torch.load(args.pretrained_model))
		print(f"Loaded pretrained model from {args.pretrained_model}")
	return model 

def create_optimizer(args, model):
	"""Create optimizer (not used in inference)"""
	if args.optim == 'sgd':
		return optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd)
	elif args.optim == 'adam':
		return optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
	elif args.optim == "adadelta":
		return optim.Adadelta(model.parameters())

# Initialize model and run inference
model = create_model(args)
optimizer = create_optimizer(args, model)

# Run inference without gradient computation
with torch.set_grad_enabled(False):
	test_pred = []
	test_true = []
	model.eval()

print("Running inference...")
for batch_idx, data in enumerate(testing_generator):
	# Move data to GPU
	data = data.to(device)
	
	# Skip incomplete batches
	if torch.max(data.batch) < (args.batch_size - 1):
		continue
	
	# Forward pass
	outputs = model(data.pos, data.batch, data.edge_index, data.edge_attr)
	test_preds = outputs.max(dim=1)[1]
	test_pred.append(test_preds.detach().cpu().numpy())
	
	print(f"Processed batch {batch_idx + 1}")

# Process predictions and save results
test_preds = np.concatenate(test_pred)
idx_ref = 0

print("Saving predictions...")
for i in range(len(test_list)):
	# Extract predictions for current sample
	len_data_item = len(test_list[i].x)
	slice_data = test_preds[idx_ref:(idx_ref + len_data_item)].reshape(-1, 1)
	
	# Combine original coordinates with predictions
	out_data = np.concatenate((np.asarray(test_list[i].x), slice_data), axis=1)
	
	# Save to processed directory
	output_path = f"processed/cl_{i}.npy"
	np.save(output_path, out_data)
	print(f"Saved {output_path} with shape {out_data.shape}")
	
	idx_ref += len_data_item

print("Inference completed! Results saved in processed/ directory.")