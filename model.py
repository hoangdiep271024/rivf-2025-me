import torch
import torch.nn as nn
import torch.nn as nn
import torch.nn.functional as F


class LEARNet(nn.Module):
    def __init__(self, num_classes = 5):
        super(LEARNet, self).__init__()

        # Conv-1
        self.conv1 = nn.Conv2d(3, 16, kernel_size = 3, stride = 2, padding = 1)

        # Conv-2 (4 đường dẫn song song)
        self.conv2_1 = nn.Conv2d(16, 16, kernel_size = 1, stride = 2)
        self.conv2_2 = nn.Conv2d(16, 16, kernel_size = 1, stride = 2)
        self.conv2_3 = nn.Conv2d(16, 16, kernel_size = 1, stride = 2)
        self.conv2_4 = nn.Conv2d(16, 16, kernel_size = 1, stride = 2)

        # Accretion layer (Add-1)
        self.add1 = nn.Conv2d(16, 16, kernel_size = 1)
        self.add2 = nn.Conv2d(16, 16, kernel_size = 1)

        #Conv-3 (4 đường dẫn song song)
        self.conv3_1 = nn.Conv2d(16, 32, kernel_size = 3, stride = 2, padding = 1)
        self.conv3_2 = nn.Conv2d(16, 32, kernel_size = 3, stride = 2, padding = 1)
        self.conv3_3 = nn.Conv2d(16, 32, kernel_size = 3, stride = 2, padding = 1)
        self.conv3_4 = nn.Conv2d(16, 32, kernel_size = 3, stride = 2, padding = 1)

        #Accretion Layer (Add-2)
        self.add3 = nn.Conv2d(32, 32, kernel_size = 1)
        self.add4 = nn.Conv2d(32, 32, kernel_size = 1)

        # Conv-4 (4 đường dẫn song song)
        self.conv4_1 = nn.Conv2d(32, 64, kernel_size = 5, stride = 2, padding = 2)
        self.conv4_2 = nn.Conv2d(32, 64, kernel_size = 5, stride = 2, padding = 2)
        self.conv4_3 = nn.Conv2d(32, 64, kernel_size = 5, stride = 2, padding = 2)
        self.conv4_4 = nn.Conv2d(32, 64, kernel_size = 5, stride = 2, padding = 2)

        # Concatenation Layer
        self.concat = nn.Conv2d(256, 256, kernel_size = 1)
        
        # Local Response Normalization (LRN)
        self.lrn = nn.LocalResponseNorm(256)
        
        # Conv-5
        self.conv5 = nn.Conv2d(256, 256, kernel_size = 3, stride = 2, padding = 1)

        # Dropout Layer
        #self.dropout = nn.Dropout(p = 0.5)
        
        # Fully Connected Layer
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
            x = F.relu(self.conv1(x))
            
            # Conv-2
            x1 = F.relu(self.conv2_1(x))
            x2 = F.relu(self.conv2_2(x))
            x3 = F.relu(self.conv2_3(x))
            x4 = F.relu(self.conv2_4(x))
            
            # Add-1
            x_add1 = x1 + x2
            x_add2 = x3 + x4
            
            # Conv-3
            x1 = F.relu(self.conv3_1(x1))
            x2 = F.relu(self.conv3_2(x_add1))
            x3 = F.relu(self.conv3_3(x_add2))
            x4 = F.relu(self.conv3_4(x4))
            
            # Add-2
            x_add3 = x1 + x2
            x_add4 = x3 + x4
            
            # Conv-4
            x1 = F.relu(self.conv4_1(x1))
            x2 = F.relu(self.conv4_2(x_add3))
            x3 = F.relu(self.conv4_3(x_add4))
            x4 = F.relu(self.conv4_4(x4))
            
            # Concatenation
            x = torch.cat([x1, x2, x3, x4], dim = 1)
            x = F.relu(self.concat(x))
            
            # LRN
            x = self.lrn(x)
            
            # Conv-5
            x = F.relu(self.conv5(x))
            
            # Flatten and Fully Connected
            x = F.adaptive_avg_pool2d(x, (1, 1))
            x = torch.flatten(x, 1)
            #x = self.dropout(x)
            x = self.fc(x)
            
            return x

if __name__ == "__main__":
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- init model ----
    num_classes = 5
    model = LEARNet(num_classes=num_classes).to(device)
    model.eval()
    assert model.fc.out_features == num_classes, "FC out_features != num_classes=5"
    print(f"Model ok. num_classes={num_classes}, params={sum(p.numel() for p in model.parameters())}")

    # ---- dummy input (RGB 112x112) ----
    B, C, H, W = 4, 3, 112, 112
    x = torch.randn(B, C, H, W, device=device)

    # ---- forward ----
    with torch.no_grad():
        logits = model(x)

    print(f"Input shape : {tuple(x.shape)}")
    print(f"Output shape: {tuple(logits.shape)}  # expect ({B}, {num_classes})")