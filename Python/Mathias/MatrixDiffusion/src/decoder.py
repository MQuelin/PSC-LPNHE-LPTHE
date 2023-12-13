import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention

class VAE_AttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, channels)
        self.attention = SelfAttention(1, channels)
    
    def forward(self, x):
        # x: (Batch_Size, Features, Height, Width, Depth)
        residue = x

        # (Batch_Size, Features, Height, Width, Depth) -> (Batch_Size, Features, Height, Width, Depth)
        x = self.groupnorm(x)

        n, c, h, w, d = x.shape
        
        # (Batch_Size, Features, Height, Width, Depth) -> (Batch_Size, Features, Height * Width * Depth)
        x = x.view((n, c, h * w * d))
        
        # (Batch_Size, Features, Height * Width * Depth) -> (Batch_Size, Height * Width * Depth, Features). Each pixel becomes a feature of size "Features", the sequence length is "Height * Width * Depth".
        x = x.transpose(-1, -2)
        
        # Perform self-attention WITHOUT mask
        # (Batch_Size, Height * Width * Depth, Features) -> (Batch_Size, Height * Width * Depth, Features)
        x = self.attention(x)
        
        # (Batch_Size, Height * Width * Depth, Features) -> (Batch_Size, Features, Height * Width * Depth)
        x = x.transpose(-1, -2)
        
        # (Batch_Size, Features, Height * Width * Depth) -> (Batch_Size, Features, Height, Width, Depth)
        x = x.view((n, c, h, w, d))
        
        # (Batch_Size, Features, Height, Width, Depth) + (Batch_Size, Features, Height, Width, Depth) -> (Batch_Size, Features, Height, Width, Depth) 
        x += residue

        # (Batch_Size, Features, Height, Width, Depth)
        return x
    
class VAE_ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.groupnorm_1 = nn.GroupNorm(32, in_channels)
        self.conv_1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)

        self.groupnorm_2 = nn.GroupNorm(32, out_channels)
        self.conv_2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv3d(in_channels, out_channels, kernel_size=1, padding=0)
    
    def forward(self, x):
        # x: (Batch_Size, In_Channels, Height, Width, Depth)

        residue = x

        # (Batch_Size, In_Channels, Height, Width, Depth) -> (Batch_Size, In_Channels, Height, Width, Depth)
        x = self.groupnorm_1(x)
        
        # (Batch_Size, In_Channels, Height, Width, Depth) -> (Batch_Size, In_Channels, Height, Width, Depth)
        x = F.silu(x)
        
        # (Batch_Size, In_Channels, Height, Width, Depth) -> (Batch_Size, Out_Channels, Height, Width, Depth)
        x = self.conv_1(x)
        
        # (Batch_Size, Out_Channels, Height, Width, Depth) -> (Batch_Size, Out_Channels, Height, Width, Depth)
        x = self.groupnorm_2(x)
        
        # (Batch_Size, Out_Channels, Height, Width, Depth) -> (Batch_Size, Out_Channels, Height, Width, Depth)
        x = F.silu(x)
        
        # (Batch_Size, Out_Channels, Height, Width, Depth) -> (Batch_Size, Out_Channels, Height, Width, Depth)
        x = self.conv_2(x)
        
        # (Batch_Size, Out_Channels, Height, Width, Depth) -> (Batch_Size, Out_Channels, Height, Width, Depth)
        return x + self.residual_layer(residue)

class VAE_Decoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            # (Batch_Size, 4, Height / 8, Width / 8, Depth / 8) -> (Batch_Size, 4, Height / 8, Width / 8, Depth / 8)
            nn.Conv3d(4, 4, kernel_size=1, padding=0),

            # (Batch_Size, 4, Height / 8, Width / 8, Depth / 8) -> (Batch_Size, 512, Height / 8, Width / 8, Depth / 8)
            nn.Conv3d(4, 512, kernel_size=3, padding=1),
            
            # (Batch_Size, 512, Height / 8, Width / 8, Depth / 8) -> (Batch_Size, 512, Height / 8, Width / 8, Depth / 8)
            VAE_ResidualBlock(512, 512), 
            
            # (Batch_Size, 512, Height / 8, Width / 8, Depth / 8) -> (Batch_Size, 512, Height / 8, Width / 8, Depth / 8)
            VAE_AttentionBlock(512), 
            
            # (Batch_Size, 512, Height / 8, Width / 8, Depth / 8) -> (Batch_Size, 512, Height / 8, Width / 8, Depth / 8)
            VAE_ResidualBlock(512, 512), 
            
            # (Batch_Size, 512, Height / 8, Width / 8, Depth / 8) -> (Batch_Size, 512, Height / 8, Width / 8, Depth / 8)
            VAE_ResidualBlock(512, 512), 
            
            # (Batch_Size, 512, Height / 8, Width / 8, Depth / 8) -> (Batch_Size, 512, Height / 8, Width / 8, Depth / 8)
            VAE_ResidualBlock(512, 512), 
            
            # (Batch_Size, 512, Height / 8, Width / 8, Depth / 8) -> (Batch_Size, 512, Height / 8, Width / 8, Depth / 8)
            VAE_ResidualBlock(512, 512), 
            
            # Repeats the rows and columns of the data by scale_factor (like when you resize an image by doubling its size).
            # (Batch_Size, 512, Height / 8, Width / 8, Depth / 8) -> (Batch_Size, 512, Height / 4, Width / 4, Depth / 4)
            nn.Upsample(scale_factor=2),
            
            # (Batch_Size, 512, Height / 4, Width / 4, Depth / 4) -> (Batch_Size, 512, Height / 4, Width / 4, Depth / 4)
            nn.Conv3d(512, 512, kernel_size=3, padding=1), 
            
            # (Batch_Size, 512, Height / 4, Width / 4, Depth / 4) -> (Batch_Size, 512, Height / 4, Width / 4, Depth / 4)
            VAE_ResidualBlock(512, 512), 
            
            # (Batch_Size, 512, Height / 4, Width / 4, Depth / 4) -> (Batch_Size, 512, Height / 4, Width / 4, Depth / 4)
            VAE_ResidualBlock(512, 512), 
            
            # (Batch_Size, 512, Height / 4, Width / 4, Depth / 4) -> (Batch_Size, 512, Height / 4, Width / 4, Depth / 4)
            VAE_ResidualBlock(512, 512), 
            
            # (Batch_Size, 512, Height / 4, Width / 4, Depth / 4) -> (Batch_Size, 512, Height / 2, Width / 2, Depth / 2)
            nn.Upsample(scale_factor=2), 
            
            # (Batch_Size, 512, Height / 2, Width / 2, Depth / 2) -> (Batch_Size, 512, Height / 2, Width / 2, Depth / 2)
            nn.Conv3d(512, 512, kernel_size=3, padding=1), 
            
            # (Batch_Size, 512, Height / 2, Width / 2, Depth / 2) -> (Batch_Size, 256, Height / 2, Width / 2, Depth / 2)
            VAE_ResidualBlock(512, 256), 
            
            # (Batch_Size, 256, Height / 2, Width / 2, Depth / 2) -> (Batch_Size, 256, Height / 2, Width / 2, Depth / 2)
            VAE_ResidualBlock(256, 256), 
            
            # (Batch_Size, 256, Height / 2, Width / 2, Depth / 2) -> (Batch_Size, 256, Height / 2, Width / 2, Depth / 2)
            VAE_ResidualBlock(256, 256), 
            
            # (Batch_Size, 256, Height / 2, Width / 2, Depth / 2) -> (Batch_Size, 256, Height, Width, Depth)
            nn.Upsample(scale_factor=2), 
            
            # (Batch_Size, 256, Height, Width, Depth) -> (Batch_Size, 256, Height, Width, Depth)
            nn.Conv3d(256, 256, kernel_size=3, padding=1), 
            
            # (Batch_Size, 256, Height, Width, Depth) -> (Batch_Size, 128, Height, Width, Depth)
            VAE_ResidualBlock(256, 128), 
            
            # (Batch_Size, 128, Height, Width, Depth) -> (Batch_Size, 128, Height, Width, Depth)
            VAE_ResidualBlock(128, 128), 
            
            # (Batch_Size, 128, Height, Width, Depth) -> (Batch_Size, 128, Height, Width, Depth)
            VAE_ResidualBlock(128, 128), 
            
            # (Batch_Size, 128, Height, Width, Depth) -> (Batch_Size, 128, Height, Width, Depth)
            nn.GroupNorm(32, 128), 
            
            # (Batch_Size, 128, Height, Width, Depth) -> (Batch_Size, 128, Height, Width, Depth)
            nn.SiLU(), 
            
            # (Batch_Size, 128, Height, Width, Depth) -> (Batch_Size, 1, Height, Width, Depth)
            nn.Conv3d(128, 1, kernel_size=3, padding=1), 
        )

    def forward(self, x):
        # x: (Batch_Size, 4, Height / 8, Width / 8, Depth / 8)
        
        # Remove the scaling added by the Encoder.
        x /= 0.18215

        for module in self:
            x = module(x)

        # (Batch_Size, 3, Height, Width, Depth)
        return x