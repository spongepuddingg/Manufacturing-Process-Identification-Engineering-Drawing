import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_sequence
from torch.utils.data import DataLoader
from torchvision import transforms
import os
import numpy as np
# from dataset import graphData
# from sketchData import *
from clusterData import *
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


def get_angle(x1, x2, y1, y2, x3, x4, y3, y4):
	m1 = (y2 - y1)/(x2 - x1)
	m2 = (y4 - y3)/(x4 - x3)
	k = (m1-m2)/(1+m1*m2)
	if k < 0:
		k = -1 * k
	a = np.arctan(k)
	return a

def cluster_collate(batch):
	# batch contains a list of tuples of structure (sequence, target)
	#print('batch list size', len(batch))
	data = [item[0] for item in batch]
	data_length = [len(sq) for sq in data]
	cnt = 0
	gcn_batch = []
	for length in data_length:
		y = [cnt] * length
		gcn_batch.extend(y)
		cnt += 1
	#print('gcn batch length', len(gcn_batch))
	gcn_batch = torch.tensor(gcn_batch, dtype=torch.long)
	data = pack_sequence(data, enforce_sorted=False)
	targets = [item[1] for item in batch]
	targets = pack_sequence(targets, enforce_sorted=False).data 
	targets = targets.long()

	data_orin = [item[2] for item in batch]
	data_orin = pack_sequence(data_orin, enforce_sorted=False)

	# calculate edge attributes
	knn = DilatedKnnGraph(3, 1, True, 0.3)
	edge_index = knn(data.data, gcn_batch)
	edge_attr = []
	for i in range(edge_index.shape[1]):
		idx1, idx2 = edge_index[0, i].item(), edge_index[1, i].item()
		#print('indices', idx1, idx2)
		x1, x2, y1, y2 = data.data[idx1, :] 
		x3, x4, y3, y4 = data.data[idx2, :]
		angle = get_angle(x1, x2, y1, y2, x3, x4, y3, y4)
		if math.isnan(angle):
			angle = 0
		#print('angle', angle)
		edge_attr.append(angle)
	edge_attr = torch.tensor(edge_attr, dtype=torch.float)
	edge_attr = edge_attr.view(-1, 1)
	return [data, gcn_batch, edge_index, edge_attr, targets, data_orin]



