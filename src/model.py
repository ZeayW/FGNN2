import torch as th
import torch.nn as nn
import torch.nn.functional as F

class Classifier(nn.Module):
    def __init__(
        self, GCN,mlp
    ):
        super(Classifier, self).__init__()

        self.encoder = GCN
        self.readout = mlp

        # print(self.layers)


class BiClassifier(nn.Module):
    def __init__(
        self, GCN1,GCN2,mlp,flag_usage='local'
    ):
        super(BiClassifier, self).__init__()

        self.GCN1 = GCN1
        self.GCN2 = GCN2
        self.mlp=mlp
        self.flag_usage = flag_usage
        # print(self.layers)
    def forward(self, in_blocks, in_features,out_blocks,out_features):
        if self.GCN2 is None:
            h = self.GCN1(in_blocks,in_features,flag_usage=self.flag_usage)
        elif self.GCN1 is None:
            h = self.GCN2(out_blocks, out_features,flag_usage=self.flag_usage)
        else:
            h = self.GCN1(in_blocks, in_features,flag_usage=self.flag_usage)
            rh = self.GCN2(out_blocks,out_features,flag_usage=self.flag_usage)
            h = th.cat((h,rh),1)
        h = self.mlp(h)

        return h