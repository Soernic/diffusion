import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace

class TNet(nn.Module):
    def __init__(self, k):
        super(TNet, self).__init__()
        self.k = k
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k*k)
        
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.size()[0]

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = torch.eye(self.k, requires_grad=True).repeat(batchsize, 1, 1).to(x.device)
        x = x.view(-1, self.k, self.k) + iden
        return x

class PointNetWithTimeEmbedding(nn.Module):
    def __init__(self, k=2, feature_transform=False):
        super(PointNetWithTimeEmbedding, self).__init__()
        self.feature_transform = feature_transform
        self.tnet1 = TNet(k=6)
        self.conv1 = nn.Conv1d(6, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)
        
        if self.feature_transform:
            self.tnet2 = TNet(k=64)

    def forward(self, x, beta):
        # set_trace()
        # Fix shape errors for sampling
        x = x.permute(0, 2, 1) if x.size()[2] == 3 else x

        n_pts = x.size()[2]
        
        # Create time embedding
        time_embedding = torch.stack([beta, torch.sin(beta), torch.cos(beta)], dim=-1).unsqueeze(1).repeat(1, n_pts, 1).permute(0, 2, 1)
        # time_embedding = torch.stack([beta, torch.sin(beta), torch.cos(beta)], dim=-1).unsqueeze(1).repeat(1, 1, n_pts).permute(0, 2, 1)
        x = torch.cat((x, time_embedding), dim=1) # [batch_size, 6, 2048]

        trans = self.tnet1(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)

        x = F.relu(self.bn1(self.conv1(x)))
                
        if self.feature_transform:
            trans_feat = self.tnet2(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)
        else:
            trans_feat = None
        

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)


        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)
        return x, trans_feat

    def feature_transform_regularizer(self, trans):
        d = trans.size()[1]
        I = torch.eye(d, requires_grad=True).repeat(trans.size()[0], 1, 1).to(trans.device)
        loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - I, dim=(1, 2)))
        return loss


