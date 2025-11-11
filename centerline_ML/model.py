"""DeepGCN Model for Centerline Detection

This module implements a Deep Graph Convolutional Network for classifying
line segments in technical drawings as centerlines.
"""

import torch
from torch.nn import Linear as Lin
import torch_geometric as tg
from gcn_lib.sparse import MultiSeq, MLP, GraphConv, PlainDynBlock, ResDynBlock, DenseDynBlock, DilatedKnnGraph
from torch_scatter import scatter

class DeepGCN(torch.nn.Module):
    """Deep Graph Convolutional Network for centerline classification
    
    Args:
        opt: Configuration object containing model hyperparameters
    """
    
    def __init__(self, opt):
        super(DeepGCN, self).__init__()
        # Extract hyperparameters
        channels = opt.n_filters
        k = opt.k
        act = opt.act
        norm = opt.norm
        bias = opt.bias
        epsilon = opt.epsilon
        stochastic = opt.stochastic
        conv = opt.conv
        c_growth = channels

        self.n_blocks = opt.n_blocks

        # Initial graph convolution layer
        self.head = GraphConv(opt.in_channels, channels, conv, act, norm, bias)
        if opt.block.lower() == 'res':
            self.backbone = MultiSeq(*[ResDynBlock(channels, k, 1, conv, act, norm, bias, stochastic=stochastic, epsilon=epsilon)
                                       for i in range(self.n_blocks-1)])
            fusion_dims = int(channels + c_growth * (self.n_blocks - 1))
        elif opt.block.lower() == 'dense':
            self.backbone = MultiSeq(*[DenseDynBlock(channels+c_growth*i, c_growth, k, 1,
                                                     conv, act, norm, bias, stochastic=stochastic, epsilon=epsilon)
                                       for i in range(self.n_blocks-1)])
            fusion_dims = int(
                (channels + channels + c_growth * (self.n_blocks - 1)) * self.n_blocks // 2)
        else:
            # Use PlainGCN without skip connection and dilated convolution.
            stochastic = False
            self.backbone = MultiSeq(
                *[PlainDynBlock(channels, k, 1, conv, act, norm, bias, stochastic=stochastic, epsilon=epsilon)
                  for i in range(self.n_blocks - 1)])
            fusion_dims = int(channels + c_growth * (self.n_blocks - 1))
        self.fusion_block = MLP([fusion_dims, 1024], act, norm, bias)
        self.prediction = MultiSeq(*[MLP([fusion_dims, 512], act, norm, bias),
                                     MLP([512, 256], act, norm, bias, drop=opt.dropout),
                                     MLP([256, opt.n_classes], None, None, bias)])
        self.model_init()

    def model_init(self):
        for m in self.modules():
            if isinstance(m, Lin):
                torch.nn.init.kaiming_normal_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True

    def forward(self, x, data_length, edge_index, edge_attr):
        """Forward pass through the DeepGCN
        
        Args:
            x: Node features (line segment coordinates)
            data_length: Batch information
            edge_index: Graph edge indices
            edge_attr: Edge attributes (angles between line segments)
            
        Returns:
            Classification logits for each line segment
        """
        # Initial feature extraction
        feats = [self.head(x, edge_index, edge_attr=edge_attr)]
        
        # Pass through backbone blocks
        for i in range(self.n_blocks-1):
            feats.append(self.backbone[i](feats[-1], data_length)[0])
        
        # Concatenate features from all layers
        feats = torch.cat(feats, dim=1)
        
        # Final classification
        return self.prediction(feats)


