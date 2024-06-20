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

        x = x / torch.sqrt(torch.tensor(x.shape[-1], dtype=torch.float32))
        x = torch.mean(x, 2, keepdim=True)  # Replaced torch.max with torch.mean
        x = x.view(batchsize, -1)  # Ensure batchsize is retained

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = torch.eye(self.k, requires_grad=True).repeat(batchsize, 1, 1).to(x.device)
        x = x.view(-1, self.k, self.k) + iden
        return x

class PointNetMeanWithTimeEmbedding(nn.Module):
    def __init__(self, k=2, feature_transform=False):
        super(PointNetMeanWithTimeEmbedding, self).__init__()
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
        x = x.permute(0, 2, 1) if x.size()[2] == 3 else x

        # Normalize x
        x_mean = x.mean(dim=2, keepdim=True)
        x_std = x.std(dim=2, keepdim=True)
        x = (x - x_mean) / (x_std + 1e-5)  # Add a small value to avoid division by zero

        n_pts = x.size()[2]
        
        time_embedding = torch.stack([beta, torch.sin(beta), torch.cos(beta)], dim=-1).unsqueeze(1).repeat(1, n_pts, 1).permute(0, 2, 1)
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
        
        # Debug: Print shape before and after mean pooling
        # print(f"Shape before mean pooling: {x.shape}")
        x = torch.mean(x, 2, keepdim=True)  # Replaced torch.max with torch.mean
        # print(f"Shape after mean pooling: {x.shape}")
        
        x = x.view(x.size(0), -1)  # Ensure the batchsize is retained
        # print(f"Shape after view: {x.shape}")
        # set_trace()

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)
        return x, trans_feat

    def feature_transform_regularizer(self, trans):
        d = trans.size()[1]
        I = torch.eye(d, requires_grad=True).repeat(trans.size()[0], 1, 1).to(trans.device)
        loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - I, dim=(1, 2)))
        return loss
