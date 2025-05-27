import torch
import torch.nn as nn
import torch.nn.functional as F

class TimeEmbedding(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim // 4, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )

    def forward(self, t):
        # Handle both single timestep and batched timesteps
        # If t is a 2D tensor already, flatten it
        if t.dim() > 1:
            t = t.view(-1)
            
        # Create sinusoidal embeddings for timesteps
        half_dim = self.embedding_dim // 8
        emb = torch.log(torch.tensor(10000.0, device=t.device)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        
        # Pass through MLP
        return self.mlp(emb)

class SelfAttention(nn.Module):
    """Self-attention module for diffusion model"""
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        assert channels % num_heads == 0, "channels must be divisible by num_heads"
        
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        
    def forward(self, x):
        b, c, h, w = x.shape
        
        # Apply normalization
        x_norm = self.norm(x)
        
        # Compute query, key, value
        qkv = self.qkv(x_norm).reshape(b, 3, self.num_heads, self.head_dim, h*w)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]  # (b, num_heads, head_dim, h*w)
        
        # Compute attention
        q = q.transpose(-1, -2)  # (b, num_heads, h*w, head_dim)
        attention = torch.matmul(q, k) * (self.head_dim ** -0.5)  # (b, num_heads, h*w, h*w)
        attention = F.softmax(attention, dim=-1)
        
        # Apply attention to values
        out = torch.matmul(attention, v.transpose(-1, -2))  # (b, num_heads, h*w, head_dim)
        out = out.transpose(-1, -2).reshape(b, c, h, w)
        
        # Final projection
        return x + self.proj(out)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_channels, dropout_rate=0.1):
        super().__init__()
        self.time_mlp = nn.Linear(time_channels, out_channels)
        
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout_rate)
        
        # Skip connection handling
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x, time_emb):
        # First conv block
        h = self.norm1(x)
        h = self.act(h)
        h = self.conv1(h)
        
        # Add time embedding
        time_emb = self.act(self.time_mlp(time_emb))
        h = h + time_emb.unsqueeze(-1).unsqueeze(-1)
        
        # Second conv block
        h = self.norm2(h)
        h = self.act(h)
        h = self.conv2(h)
        h = self.dropout(h)
        
        # Skip connection
        return h + self.shortcut(x)

class EnhancedUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_channels=64, time_emb_dim=128, use_attention=True):
        super().__init__()
        # Time embedding
        self.time_embedding = TimeEmbedding(time_emb_dim)
        
        # Initial projection
        self.in_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        
        # Define channel dimensions for clarity
        c0 = base_channels
        c1 = base_channels * 2
        c2 = base_channels * 4
        c3 = base_channels * 8
        c4 = base_channels * 8  # Keep the deepest level fixed to avoid excessive growth
        
        # Downsampling blocks
        self.down1 = ConvBlock(c0, c1, time_emb_dim)
        self.down2 = ConvBlock(c1, c2, time_emb_dim)
        self.down3 = ConvBlock(c2, c3, time_emb_dim)
        self.down4 = ConvBlock(c3, c4, time_emb_dim)
        
        # Downsampling operations
        self.pool1 = nn.MaxPool2d(2)
        self.pool2 = nn.MaxPool2d(2)
        self.pool3 = nn.MaxPool2d(2)
        self.pool4 = nn.MaxPool2d(2)
        
        # Middle blocks with attention
        self.middle1 = ConvBlock(c4, c4, time_emb_dim)
        self.middle_attn = SelfAttention(c4) if use_attention else nn.Identity()
        self.middle2 = ConvBlock(c4, c4, time_emb_dim)
        
        # Upsampling blocks
        self.up1 = ConvBlock(c4 + c4, c3, time_emb_dim)  # Skip from down4
        self.up1_attn = SelfAttention(c3) if use_attention else nn.Identity()
        
        self.up2 = ConvBlock(c3 + c3, c2, time_emb_dim)  # Skip from down3
        self.up2_attn = SelfAttention(c2) if use_attention else nn.Identity()
        
        self.up3 = ConvBlock(c2 + c2, c1, time_emb_dim)  # Skip from down2
        self.up4 = ConvBlock(c1 + c1, c0, time_emb_dim)  # Skip from down1
        
        # Final output
        self.out_norm = nn.GroupNorm(8, c0)
        self.out_act = nn.SiLU()
        self.out_conv = nn.Conv2d(c0, out_channels, 3, padding=1)
    
    def forward(self, x, t):
        # Time embedding
        t_emb = self.time_embedding(t)
        
        # Initial projection
        x0 = self.in_conv(x)
        
        # Downsampling path with skip connections
        x1 = self.down1(x0, t_emb)
        x1_pool = self.pool1(x1)
        
        x2 = self.down2(x1_pool, t_emb)
        x2_pool = self.pool2(x2)
        
        x3 = self.down3(x2_pool, t_emb)
        x3_pool = self.pool3(x3)
        
        x4 = self.down4(x3_pool, t_emb)
        x4_pool = self.pool4(x4)
        
        # Middle blocks with attention
        x_middle = self.middle1(x4_pool, t_emb)
        x_middle = self.middle_attn(x_middle)
        x_middle = self.middle2(x_middle, t_emb)
        
        # Upsampling path with skip connections
        x_up1 = F.interpolate(x_middle, size=x4.shape[2:], mode='nearest')
        x_up1 = torch.cat([x_up1, x4], dim=1)  # Skip connection
        x_up1 = self.up1(x_up1, t_emb)
        x_up1 = self.up1_attn(x_up1)
        
        x_up2 = F.interpolate(x_up1, size=x3.shape[2:], mode='nearest')
        x_up2 = torch.cat([x_up2, x3], dim=1)  # Skip connection
        x_up2 = self.up2(x_up2, t_emb)
        x_up2 = self.up2_attn(x_up2)
        
        x_up3 = F.interpolate(x_up2, size=x2.shape[2:], mode='nearest')
        x_up3 = torch.cat([x_up3, x2], dim=1)  # Skip connection
        x_up3 = self.up3(x_up3, t_emb)
        
        x_up4 = F.interpolate(x_up3, size=x1.shape[2:], mode='nearest')
        x_up4 = torch.cat([x_up4, x1], dim=1)  # Skip connection
        x_up4 = self.up4(x_up4, t_emb)
        
        # Final output
        x_out = self.out_norm(x_up4)
        x_out = self.out_act(x_out)
        x_out = self.out_conv(x_out)
        
        return x_out

# Keep the SimpleUNet for backwards compatibility
class SimpleUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_channels=32, time_emb_dim=64):
        super().__init__()
        # Time embedding
        self.time_embedding = TimeEmbedding(time_emb_dim)
        
        # Initial projection
        self.in_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        
        # Define channel dimensions for clarity
        c0 = base_channels
        c1 = base_channels * 2
        c2 = base_channels * 4
        
        # Downsampling blocks
        self.down1 = ConvBlock(c0, c1, time_emb_dim)
        self.down2 = ConvBlock(c1, c2, time_emb_dim)
        
        # Downsampling operations
        self.pool1 = nn.MaxPool2d(2)
        self.pool2 = nn.MaxPool2d(2)
        
        # Middle block
        self.middle = ConvBlock(c2, c2, time_emb_dim)
        
        # Upsampling blocks
        self.up1 = ConvBlock(c2 + c2, c1, time_emb_dim)  # Skip from down2
        self.up2 = ConvBlock(c1 + c1, c0, time_emb_dim)  # Skip from down1
        
        # Final output
        self.out_conv = nn.Conv2d(c0, out_channels, 3, padding=1)
    
    def forward(self, x, t):
        # Time embedding
        t_emb = self.time_embedding(t)
        
        # Initial projection
        x0 = self.in_conv(x)
        
        # Downsampling path with skip connections
        x1 = self.down1(x0, t_emb)
        x1_pool = self.pool1(x1)
        
        x2 = self.down2(x1_pool, t_emb)
        x2_pool = self.pool2(x2)
        
        # Middle block
        x_middle = self.middle(x2_pool, t_emb)
        
        # Upsampling path with skip connections
        x_up1 = F.interpolate(x_middle, size=x2.shape[2:], mode='nearest')
        x_up1 = torch.cat([x_up1, x2], dim=1)  # Skip connection
        x_up1 = self.up1(x_up1, t_emb)
        
        x_up2 = F.interpolate(x_up1, size=x1.shape[2:], mode='nearest')
        x_up2 = torch.cat([x_up2, x1], dim=1)  # Skip connection
        x_up2 = self.up2(x_up2, t_emb)
        
        # Final output
        return self.out_conv(x_up2)

class DiffusionModel:
    def __init__(self, model, beta_min=1e-4, beta_max=0.02, num_timesteps=1000, device='cuda'):
        self.model = model
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.num_timesteps = num_timesteps
        self.device = device
        
        # Linear beta schedule
        self.betas = torch.linspace(beta_min, beta_max, num_timesteps, device=device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)
        
        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_log_variance_clipped = torch.log(self.posterior_variance.clamp(min=1e-20))
        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)
    
    def q_sample(self, x_0, t, noise=None):
        """Forward diffusion process: q(x_t | x_0)"""
        if noise is None:
            noise = torch.randn_like(x_0)
            
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        
        return sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise
    
    def p_losses(self, x_0, t, noise=None):
        """Training loss for diffusion model"""
        if noise is None:
            noise = torch.randn_like(x_0)
            
        x_t = self.q_sample(x_0, t, noise)
        predicted_noise = self.model(x_t, t)
        
        # Simple noise prediction loss (L2)
        loss = F.mse_loss(predicted_noise, noise)
        return loss
    
    @torch.no_grad()
    def p_sample(self, x_t, t):
        """Single step of the reverse diffusion process"""
        # Predict noise - don't unsqueeze t here, let the time embedding handle it
        predicted_noise = self.model(x_t, t)
        
        # Calculate the mean for the posterior q(x_{t-1} | x_t, x_0)
        # First, estimate x_0 from the model output
        sqrt_recip_alphas_cumprod_t = self.sqrt_recip_alphas_cumprod[t]
        sqrt_recipm1_alphas_cumprod_t = self.sqrt_recipm1_alphas_cumprod[t]
        
        # Estimate x_0
        predicted_x0 = (sqrt_recip_alphas_cumprod_t.reshape(-1, 1, 1, 1) * x_t - 
                        sqrt_recipm1_alphas_cumprod_t.reshape(-1, 1, 1, 1) * predicted_noise)
        
        # Calculate posterior mean
        posterior_mean = (self.posterior_mean_coef1[t].reshape(-1, 1, 1, 1) * predicted_x0 + 
                          self.posterior_mean_coef2[t].reshape(-1, 1, 1, 1) * x_t)
        
        # Sample from posterior
        noise = torch.randn_like(x_t) if t[0] > 0 else torch.zeros_like(x_t)
        variance = torch.exp(0.5 * self.posterior_log_variance_clipped[t])
        x_t_prev = posterior_mean + variance.reshape(-1, 1, 1, 1) * noise
        
        return x_t_prev
    
    @torch.no_grad()
    def sample(self, batch_size=16, img_size=28):
        """Generate samples from noise"""
        # Start from random noise
        x = torch.randn(batch_size, 1, img_size, img_size, device=self.device)
        
        # Iteratively denoise
        for t in reversed(range(self.num_timesteps)):
            t_tensor = torch.full((batch_size,), t, device=self.device, dtype=torch.long)
            x = self.p_sample(x, t_tensor)
            
        return x 