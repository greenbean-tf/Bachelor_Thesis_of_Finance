import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, cnn, proj):
        super(ResidualBlock, self).__init__()
        self.cnn = cnn
        self.proj = proj

    def forward(self, x):
        return self.cnn(x) + self.proj(x)

class CNN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dropout=True):
        super(CNN, self).__init__()
        layers = []
        layer1 = [
            nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding),
            nn.BatchNorm1d(out_channels, track_running_stats=False),
            nn.PReLU()
        ]
        layer2 = [
            nn.Conv1d(out_channels, out_channels, kernel_size, padding='same'),
            nn.BatchNorm1d(out_channels, track_running_stats=False)
        ]
        layers += layer1
        if dropout:
            layers.append(nn.Dropout())
        layers += layer2
        self.cnn = nn.Sequential(*layers)

    def forward(self, x):
        return self.cnn(x)

class Proj(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(Proj, self).__init__()
        if in_channels == out_channels:
            self.proj = nn.Identity()
        else:
            self.proj = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride),
                nn.BatchNorm1d(out_channels, track_running_stats=False)
            )

    def forward(self, x):
        return self.proj(x)

class ResNetModule(nn.Module):
    def __init__(self):
        super(ResNetModule, self).__init__()
        self.residual_block1 = ResidualBlock(
            cnn=CNN(in_channels=3, out_channels=128, kernel_size=3, stride=2, padding=1),
            proj=Proj(in_channels=3, out_channels=128, kernel_size=1, stride=2)
        )
        self.prelu1 = nn.PReLU()
        self.residual_block2 = ResidualBlock(
            cnn=CNN(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
            proj=Proj(in_channels=128, out_channels=256, kernel_size=1, stride=2)
        )
        self.prelu2 = nn.PReLU()
        self.residual_block3 = ResidualBlock(
            cnn=CNN(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1),
            proj=Proj(in_channels=256, out_channels=512, kernel_size=1, stride=2)
        )
        self.prelu3 = nn.PReLU()
        self.residual_block4 = ResidualBlock(
            cnn=CNN(in_channels=512, out_channels=1024, kernel_size=3, stride=2, padding=1),
            proj=Proj(in_channels=512, out_channels=1024, kernel_size=1, stride=2)
        )
        self.prelu4 = nn.PReLU()
        self.maxpool = nn.MaxPool1d(3, 2, 1)
        

    def forward(self, x):
        x = self.residual_block1(x)
        x = self.prelu1(x)
        x = self.residual_block2(x)
        x = self.prelu2(x)
        x = self.residual_block3(x)
        x = self.prelu3(x)
        x = self.residual_block4(x)
        x = self.prelu4(x)
        x = self.maxpool(x)
        return x

class TransEncModule(nn.Module):
    def __init__(self):
        super(TransEncModule, self).__init__()
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(256, 4, 64, batch_first=True, dropout=0.5),8)
        self.avgpool = nn.AvgPool1d(4)

    def forward(self, x):
        x = self.transformer_encoder(x)
        x = self.avgpool(x)
        return x