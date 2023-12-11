
import torch.nn as nn
import torch.nn.functional as F

class Pt_fc(nn.Module):
    def __init__(self, args, output_channels=10):
        super(Pt_fc, self).__init__()
        self.args = args
        
        self.linear1 = nn.Linear(2048, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x):
        x = F.relu(self.bn6(self.linear1(x)))
        x = self.dp1(x)
        x = F.relu(self.bn7(self.linear2(x)))
        #x = F.normalize(x,1)
        return x
    
###########################semi-supervised network#####################################

class Semi_pt(nn.Module):
    def __init__(self, args, num_classes=10):
        super(Semi_pt, self).__init__()
        self.num_classes = num_classes
        self.branch1 = Pt_fc(args,self.num_classes)
        self.branch2 = Pt_fc(args,self.num_classes)
        self.head1 = nn.Sequential(*[nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, self.num_classes)])
        self.head2 = nn.Sequential(*[nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, self.num_classes)])
    def forward(self, data, step=1):
        if step == 1:
            pt_feat1 = self.branch1(data)
            pt_pred1 = self.head1(pt_feat1)
            return pt_feat1, pt_pred1
        elif step == 2:
            pt_feat2 = self.branch2(data)
            pt_pred2 = self.head2(pt_feat2)
            return pt_feat2, pt_pred2