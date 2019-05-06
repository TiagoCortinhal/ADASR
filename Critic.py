from torch import nn

##TODO see4c1 <- host cluster

class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.main_module = nn.Sequential(
            #nn.ReplicationPad2d(2),
            nn.Conv2d(in_channels=3, out_channels=128, kernel_size=4, stride=2, padding=1, bias=False),
            #nn.InstanceNorm2d(128, affine=True),
            nn.LeakyReLU(0.2),

            #nn.ReplicationPad2d(2),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1, bias=False),
            #nn.InstanceNorm2d(256, affine=True),
            nn.LeakyReLU(0.2),

            #nn.ReplicationPad2d(2),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1, bias=False),
            #nn.InstanceNorm2d(512, affine=True),
            nn.LeakyReLU(0.2),

            #nn.ReplicationPad2d(2),
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=4, stride=2, padding=1, bias=False),
            #nn.InstanceNorm2d(1024, affine=True),
            nn.LeakyReLU(0.2),

            #nn.ReplicationPad2d(2),
            nn.Conv2d(in_channels=1024, out_channels=2048, kernel_size=4, stride=2, padding=1, bias=False),
            #nn.InstanceNorm2d(2048, affine=True),
            nn.LeakyReLU(0.2)

        )

        self.output = nn.Sequential(
            nn.Conv2d(in_channels=2048, out_channels=1, kernel_size=4, stride=2, padding=1, bias=False)
        )

    def forward(self, x):
        x = self.main_module(x)
        res = self.output(x)
        return res