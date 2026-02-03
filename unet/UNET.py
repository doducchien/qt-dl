import torch.nn as nn
import torch



class DoubleConv(nn.Module):
    def __init__(self, in_channels:int, out_channels: int):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x:torch.Tensor):
        return  self.seq(x)


class Down(nn.Module):
    def __init__(self, in_channels:int, out_channels:int):
        super().__init__()
        self.seq = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(in_channels=in_channels, out_channels=out_channels)
        )

    def forward(self, x:torch.Tensor):
        return self.seq(x)

class Up(nn.Module):
    def __init__(self, in_channels:int, out_channels:int):
        super().__init__()
        self.seq = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=2, stride=2),
            DoubleConv(in_channels=in_channels, out_channels=out_channels)
        )

    def forward(self, x_encoder:torch.Tensor, x:torch.Tensor):
        print(x_encoder.shape, x.shape)
        new_x = torch.concat((x_encoder, x), dim=1)
        return self.seq(new_x)

class OutConv(nn.Module):
    def __init__(self, in_channels:int, out_channels:int):
        super().__init__()
        self.out = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1)

    def forward(self, x:torch.Tensor):
        return self.out(x)


class UNET(nn.Module):
    def __init__(self, n_channels:int, n_class:int):
        super().__init__()
        self.inc = DoubleConv(in_channels=n_channels, out_channels=64)
        self.down1 = Down(in_channels=64, out_channels=128)
        self.down2 = Down(in_channels=128, out_channels=256)
        self.down3 = Down(in_channels=256, out_channels=512)
        self.down4 = Down(in_channels=512, out_channels=1024)

        self.up1 = Up(in_channels=1024, out_channels=512)
        self.up2 = Up(in_channels=512, out_channels=256)
        self.up3 = Up(in_channels=256, out_channels=128)
        self.up4 = Up(in_channels=128, out_channels=64)
        self.out = OutConv(in_channels=64, out_channels=n_class)


    def forward(self, x:torch.Tensor):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x4, x5)
        x = self.up2(x3, x)
        x = self.up3(x2, x)
        x = self.up4(x1, x)
        logits = self.out(x)
        return logits

