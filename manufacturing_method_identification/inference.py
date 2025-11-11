"""Manufacturing Method Identification Inference Script

Runs inference on trained classifier to predict manufacturing methods
(Lathe, Sheet Metal, Milling) from technical drawings.
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

# Optimization arguments
parser = argparse.ArgumentParser(description='Shape Analysis')
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
parser.add_argument('--pretrained_model', type=str, help='path to pretrained model(default: none)', default='/media/louise/ubuntu_work2/lost+found/SA_classification/saved_model/best_test.pt')
parser.add_argument('--in_channels', default=5, type=int, help='')
parser.add_argument('--hidden_channels', default=256, type=int, help='')
parser.add_argument('--feat_dims', default=128, type=int, help='')
parser.add_argument('--out_channels', default=236, type=int, help='')

# dilated knn
parser.add_argument('--epsilon', default=0.2, type=float, help='stochastic epsilon for gcn')
parser.add_argument('--stochastic', default=True,  type=bool, help='stochastic for gcn, True or False')
parser.add_argument('--use_dilation', default=True,  type=bool, help='dilation')

args = parser.parse_args()
torch.autograd.set_detect_anomaly(True)

# CUDA for pytorch
use_cuda=torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

# Parameters
max_epochs = 100

# criterion
criterion = torch.nn.CrossEntropyLoss().to(device)

###### load cluster training data
###### load labels from pickle
from glob import glob
data_vars = {}
labels = pd.read_pickle("data/labels.pkl")
for filename in glob('data/Lathe/*.np[yz]'):
	data_vars[filename] = np.load(filename)
for filename in glob('data/SheetMetal/*.np[yz]'):
	data_vars[filename] = np.load(filename)
for filename in glob('data/Milling/*.np[yz]'):
	data_vars[filename] = np.load(filename)	

names = list(data_vars.keys())
test_name = names[300:400]
data = np.array(list(data_vars.values()))
test_data = data[300:400]

testing_set = sketchData(test_data, test_name, labels, transform = None) 
print("size of testing", len(testing_set))

# dataloader
test_loader = DataLoader(testing_set, batch_size = args.batch_size, shuffle = False)

# Initialize model and load pretrained weights
model = model.Classifier(args.in_channels, args.hidden_channels, args.out_channels, args.feat_dims, args.n_classes).to(device)

if args.pretrained_model:
	model.load_state_dict(torch.load(args.pretrained_model))
	print(f"Loaded pretrained model from {args.pretrained_model}")
else:
	print("Warning: No pretrained model specified!")

optimizer = create_optimizer(args, model)

# Run inference without gradient computation
with torch.set_grad_enabled(False):
	test_true = 0.0
	model.eval()
	count = 0.0

test_pred = []
class_names = ['Lathe', 'Sheet Metal', 'Milling']

print("Running inference...")
for i, test_data in enumerate(test_loader):
	# Move data to GPU
	test_inputs = test_data
	test_labels = test_data['y'].to(device).long()
	test_inputs = test_inputs.to(device)
	batch_size = test_labels.shape[0]
	
	# Forward pass
	test_outputs, prediction, link_loss, ent_loss = model(test_inputs)
	test_preds = test_outputs.max(dim=1)[1]
	test_pred.append(test_preds.detach().cpu().numpy())
	
	print(f"Processed batch {i + 1}/{len(test_loader)}")

# Convert predictions to class names
all_predictions = np.concatenate(test_pred)
predicted_classes = [class_names[pred] for pred in all_predictions]

print("\nPrediction Results:")
for i, pred_class in enumerate(predicted_classes):
	print(f"Sample {i + 1}: {pred_class}")

print(f"\nSummary:")
for class_name in class_names:
	count = predicted_classes.count(class_name)
	print(f"{class_name}: {count} samples ({count/len(predicted_classes)*100:.1f}%)")

