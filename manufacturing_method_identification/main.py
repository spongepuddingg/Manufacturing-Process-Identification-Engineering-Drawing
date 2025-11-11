"""Manufacturing Method Identification Training Script

Trains a graph neural network to classify manufacturing drawings into
three categories: Lathe, Sheet Metal, and Milling operations.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.autograd import Variable
import os
import numpy as np
from clusterData import *
from model import *
import argparse
import time
import PIL
from math import *
import numpy as np
import math
import json
from learning_helper import *
import pandas as pd 
import classifier as model
from tqdm import tqdm

# Model configuration constants
max_nodes = 400          # Maximum nodes per graph
num_class = 3           # Number of manufacturing methods (Lathe, Sheet Metal, Milling)
in_channels = 5         # Input feature dimensions
hidden_channels = 256   # Hidden layer size
feat_dims = 128        # Feature dimensions after LSTM
out_channels = 236     # Output channels before classification
num_epoches = 1        # Number of training epochs
learning_rate = 0.001  # Learning rate

# Command line argument parser
parser = argparse.ArgumentParser(description='Manufacturing Method Classification Training')
parser.add_argument('--wd', default=2e-4, type=float, help='Weight decay')
parser.add_argument('--lr', default=5e-4, type=float, help='Initial learning rate')
parser.add_argument('--lr_decay', default=0.7, type=float, help='Multiplicative factor used on learning rate at `lr_steps`')
parser.add_argument('--lr_steps', default='[100, 150, 175]', help='List of epochs where the learning rate is decreased by `lr_decay`')
parser.add_argument('--momentum', default=0.9, type=float, help='Momentum')
parser.add_argument('--optim', default='adam', help='Optimizer: sgd|adam|adadelta')
parser.add_argument('--grad_clip', default=1, type=float, help='Element-wise clipping of gradient. If 0, does not clip')

# dataset arguments
parser.add_argument('--batch_size', default=24, type=int, help='mini-batch size (default:16)')
parser.add_argument('--n_classes', default=3, type=int, help='number of classes for the line clusters')

# model args
parser.add_argument('--pretrained_model', type=str, help='path to pretrained model(default: none)', default='')
parser.add_argument('--k', default=16, type=int, help='neighbor num (default:16)')
parser.add_argument('--block', default='res', type=str, help='graph backbone block type {plain, res, dense}')
parser.add_argument('--conv', default='edge', type=str, help='graph conv layer {edge, mr}')
parser.add_argument('--act', default='leakyrelu', type=str, help='activation layer {relu, prelu, leakyrelu}')
parser.add_argument('--norm', default='batch', type=str, help='{batch, instance} normalization')
parser.add_argument('--bias', default=True,  type=bool, help='bias of conv layer True or False')
parser.add_argument('--n_filters', default=32, type=int, help='number of channels of deep features')
parser.add_argument('--n_blocks', default=7, type=int, help='number of basic blocks')
parser.add_argument('--dropout', default=0.3, type=float, help='ratio of dropout')

# dilated knn
parser.add_argument('--epsilon', default=0.2, type=float, help='stochastic epsilon for gcn')
parser.add_argument('--stochastic', default=True,  type=bool, help='stochastic for gcn, True or False')
parser.add_argument('--use_dilation', default=True,  type=bool, help='dilation')

args = parser.parse_args()
torch.autograd.set_detect_anomaly(True)

# Setup device and training configuration
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

# Training parameters
params = {'batch_size': args.batch_size,
			'shuffle': True,
			'num_workers': 8}
max_epochs = 100

# Loss function for multi-class classification
criterion = torch.nn.CrossEntropyLoss().to(device)

# Load training data and labels
from glob import glob
data_vars = {}

# Load manufacturing method labels from pickle file
labels = pd.read_pickle("data/labels.pkl")

# Load data from all three manufacturing categories
for filename in glob('data/Lathe/*.np[yz]'):
	data_vars[filename] = np.load(filename)
for filename in glob('data/SheetMetal/*.np[yz]'):
	data_vars[filename] = np.load(filename)
for filename in glob('data/Milling/*.np[yz]'):
	data_vars[filename] = np.load(filename)

# Split data into train/test sets
names = list(data_vars.keys())
test_name = names[:200]   # First 200 samples for testing
train_name = names[200:]  # Remaining samples for training
data = np.array(list(data_vars.values()))
test_data = data[:200]
train_data = data[200:]

# Create dataset objects with graph preprocessing
training_set = sketchData(train_data, train_name, labels, transform=None, train=True) 
testing_set = sketchData(test_data, test_name, labels, transform=None) 

print(f"Training samples: {len(training_set)}")
print(f"Testing samples: {len(testing_set)}")

# Create data loaders for batch processing
train_loader = DataLoader(training_set, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(testing_set, batch_size=args.batch_size, shuffle=True)

# Initialize model and optimizer
model = model.Classifier(in_channels, hidden_channels, out_channels, feat_dims, num_class).to(device)
optimizer = create_optimizer(args, model)

# Training loop
best_testacc = 0
for epoch in range(max_epochs):
	train_true = 0.0
	model.train()
	train_loss = 0.0
	count = 0.0
	batch_num = 0
	for i, data in enumerate(tqdm(train_loader)):
		# Move data to GPU
		inputs = data
		labels = data['y'].to(device).long()
		inputs = inputs.to(device)
		batch_size = labels.shape[0]
		
		# Forward pass
		optimizer.zero_grad()
		outputs, prediction, link_loss, ent_loss = model(inputs)
		
		# Combined loss: classification + graph regularization
		loss = criterion(outputs, Variable(labels)).to(device) + link_loss + ent_loss
		
		# Backward pass
		loss.backward()
		optimizer.step()

		preds = outputs.max(dim=1)[1]
		count += batch_size
		train_loss += loss.item() * batch_size
		train_true += (preds == labels).sum().item()
		batch_num+=1
	# Print training metrics
	train_acc = train_true * 100.0 / count
	print(f'Epoch {epoch}: Loss={train_loss/count:.6f}, Train Acc={train_acc:.2f}%')
		
	if epoch % 1 == 0:
		with torch.set_grad_enabled(False):
			test_true = 0.0
			model.eval()
			#test_loss = 0.0
			count = 0.0

		for i, test_data in enumerate(test_loader):
			# Move test data to GPU
			test_inputs = test_data
			test_labels = test_data['y'].to(device).long()
			test_inputs = test_inputs.to(device)
			batch_size = test_labels.shape[0]
			
			# Forward pass for evaluation
			test_outputs, prediction, link_loss, ent_loss = model(test_inputs)
			test_preds = test_outputs.max(dim=1)[1]
			
			# Accumulate correct predictions
			count += batch_size
			test_true += (test_preds == test_labels).sum().item()
		# Calculate and print test metrics
		test_acc = test_true * 100.0 / count
		print(f'Test Epoch {epoch}: Test Acc={test_acc:.2f}%')
		
		# Save best model
		if test_acc >= best_testacc:
			best_testacc = test_acc
			torch.save(model.state_dict(), "saved_model/best_test.pt")
			print("Model saved!")

print(f"Best test accuracy: {best_testacc:.2f}%")

		