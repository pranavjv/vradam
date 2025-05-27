import torch.nn as nn
import torch.nn.functional as F
import torch
import math

    
    
class DeeperCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # Feature extractor: three convolutional blocks (each with two conv layers + pool)
        self.features = nn.Sequential(
            # Block 1: 3 → 64
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),            # 32×32 → 16×16

            # Block 2: 64 → 128
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),            # 16×16 → 8×8

            # Block 3: 128 → 256
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),            # 8×8 → 4×4
        )

        # Classifier: flatten → FC → ReLU → FC (no Dropout)
        self.classifier = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)     # flatten
        x = self.classifier(x)
        return x
    
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_encoder_layers, 
                 num_decoder_layers, dim_feedforward, dropout=0.5):
        super(TransformerModel, self).__init__()
        
        # Create embeddings for source and target sequences
        self.encoder_embedding = nn.Embedding(vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # Transformer model
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False
        )
        
        # Output layer
        self.output_layer = nn.Linear(d_model, vocab_size)
        
        # Initialize parameters
        self.d_model = d_model
        self.init_weights()
        
    def init_weights(self):
        # Initialize embeddings
        initrange = 0.1
        self.encoder_embedding.weight.data.uniform_(-initrange, initrange)
        self.decoder_embedding.weight.data.uniform_(-initrange, initrange)
        
        # Initialize output layer
        self.output_layer.bias.data.zero_()
        self.output_layer.weight.data.uniform_(-initrange, initrange)
    
    def generate_square_subsequent_mask(self, sz):
        # Generate mask for autoregressive decoding
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def forward(self, src, tgt):
        # src, tgt shapes: [seq_len, batch_size]
        
        # Create source and target masks
        src_mask = None  # All source tokens are attended to
        tgt_mask = self.generate_square_subsequent_mask(tgt.size(0)).to(tgt.device)
        
        # Embed and add positional encoding
        src_emb = self.encoder_embedding(src) * math.sqrt(self.d_model)
        src_emb = self.pos_encoder(src_emb)
        
        tgt_emb = self.decoder_embedding(tgt) * math.sqrt(self.d_model)
        tgt_emb = self.pos_encoder(tgt_emb)
        
        # Pass through transformer
        output = self.transformer(
            src_emb, tgt_emb, 
            src_mask=src_mask, 
            tgt_mask=tgt_mask
        )
        
        # Project to vocabulary size
        output = self.output_layer(output)
        
        return output
    
    
class MLPModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, dropout=0.5):
        super(MLPModel, self).__init__()
        
        # Create embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # MLP layers
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        
        # Initialize parameters
        self.init_weights()
    
    def init_weights(self):
        # Initialize embeddings
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        
        # Initialize linear layers
        self.fc1.weight.data.uniform_(-initrange, initrange)
        self.fc1.bias.data.zero_()
        self.fc2.weight.data.uniform_(-initrange, initrange)
        self.fc2.bias.data.zero_()
        
    def forward(self, x):
        # x shape: [batch_size, seq_len]
        
        # Embed tokens
        embedded = self.embedding(x)  # [batch_size, seq_len, embed_dim]
        
        # Average token embeddings
        pooled = embedded.mean(dim=1)  # [batch_size, embed_dim]
        
        # Pass through MLP
        hidden = F.relu(self.fc1(pooled))
        hidden = self.dropout(hidden)
        output = self.fc2(hidden)
        
        return output


class ContrastiveModel(nn.Module):
    """
    Model for contrastive learning tasks using the SimCLR approach.
    It consists of an encoder network (ResNet-like) followed by a projection head.
    """
    def __init__(self, embedding_dim=128, projection_dim=64):
        super(ContrastiveModel, self).__init__()
        
        # Encoder network (simplified ResNet-like structure)
        self.encoder = nn.Sequential(
            # Initial convolution block
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # ResBlock 1
            self._make_residual_block(64, 128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # ResBlock 2
            self._make_residual_block(128, 256),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # ResBlock 3
            self._make_residual_block(256, 512),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Global average pooling
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Projection head
        self.projector = nn.Sequential(
            nn.Linear(512, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embedding_dim, projection_dim)
        )
        
        # Classification head (used only for downstream tasks)
        self.classifier = nn.Linear(512, 10)  # 10 classes for CIFAR-10
        
        # Initialize weights
        self._initialize_weights()
    
    def _make_residual_block(self, in_channels, out_channels):
        """Create a residual block for the network"""
        layers = []
        
        # First convolution (potentially with downsampling)
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        
        # Second convolution
        layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))
        
        # Shortcut connection with projection if needed
        self.shortcut = None
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
            
        # Add final ReLU
        layers.append(nn.ReLU(inplace=True))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x, return_embedding=False, return_projection=True):
        """
        Forward pass through the network
        
        Args:
            x: Input tensor
            return_embedding: Whether to return the embedding
            return_projection: Whether to return the projection
            
        Returns:
            - If return_embedding and not return_projection: returns embedding
            - If return_projection and not return_embedding: returns projection
            - If both are True: returns (embedding, projection)
            - If both are False: returns classification logits
        """
        # Get embeddings from encoder
        embedding = self.encoder(x)
        embedding = embedding.view(embedding.size(0), -1)  # Flatten
        
        # Return based on flags
        if return_embedding and not return_projection:
            return embedding
        
        # Get projection
        projection = self.projector(embedding)
        
        if return_projection and not return_embedding:
            return projection
        
        if return_embedding and return_projection:
            return embedding, projection
        
        # Default: return classification logits
        logits = self.classifier(embedding)
        return logits
    
    
class GFlowNetModel(nn.Module):
    """
    GFlowNet model for generating grid-based structures.
    Learns to sample from a distribution by learning a policy that constructs objects sequentially.
    """
    def __init__(self, 
                grid_size=8, 
                hidden_dim=128, 
                num_layers=3, 
                action_dim=5,  # 4 directions + stop
                state_dim=3):  # Channels representing grid state
        super(GFlowNetModel, self).__init__()
        
        self.grid_size = grid_size
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.state_dim = state_dim
        
        # Grid encoder: processes the current grid state
        self.grid_encoder = nn.Sequential(
            nn.Conv2d(state_dim, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        
        # State processor (processes flattened state after grid encoding)
        self.state_processor = nn.Sequential(
            nn.Linear(hidden_dim * grid_size * grid_size, hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True)
        )
        
        # Forward policy network (predicts actions to take)
        policy_layers = []
        input_size = hidden_dim
        for _ in range(num_layers - 1):
            policy_layers.append(nn.Linear(input_size, hidden_dim))
            policy_layers.append(nn.ReLU(inplace=True))
            input_size = hidden_dim
        policy_layers.append(nn.Linear(hidden_dim, action_dim))
        self.policy_net = nn.Sequential(*policy_layers)
        
        # Backward policy network (for flow matching)
        backward_layers = []
        input_size = hidden_dim
        for _ in range(num_layers - 1):
            backward_layers.append(nn.Linear(input_size, hidden_dim))
            backward_layers.append(nn.ReLU(inplace=True))
            input_size = hidden_dim
        backward_layers.append(nn.Linear(hidden_dim, action_dim))
        self.backward_net = nn.Sequential(*backward_layers)
        
        # Flow network (edge flow prediction)
        flow_layers = []
        input_size = hidden_dim
        for _ in range(num_layers - 1):
            flow_layers.append(nn.Linear(input_size, hidden_dim))
            flow_layers.append(nn.ReLU(inplace=True))
            input_size = hidden_dim
        flow_layers.append(nn.Linear(hidden_dim, 1))
        self.flow_net = nn.Sequential(*flow_layers)
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def encode_state(self, grid_state):
        """
        Encode the current grid state
        
        Args:
            grid_state: Tensor of shape [batch_size, state_dim, grid_size, grid_size]
                       representing the current grid state
        
        Returns:
            state_embedding: Encoded state representation
        """
        # Encode grid
        batch_size = grid_state.shape[0]
        x = self.grid_encoder(grid_state)  # [batch_size, hidden_dim, grid_size, grid_size]
        
        # Flatten spatial dimensions
        x = x.view(batch_size, -1)  # [batch_size, hidden_dim * grid_size * grid_size]
        
        # Process state
        state_embedding = self.state_processor(x)  # [batch_size, hidden_dim]
        
        return state_embedding
    
    def forward_policy(self, grid_state):
        """
        Forward policy computes P(action|state) for constructing objects
        
        Args:
            grid_state: Current grid state 
                       [batch_size, state_dim, grid_size, grid_size]
        
        Returns:
            action_logits: Logits for each possible action
        """
        state_embedding = self.encode_state(grid_state)
        action_logits = self.policy_net(state_embedding)
        return action_logits
    
    def backward_policy(self, grid_state):
        """
        Backward policy computes P(parent_state|state) for flow matching
        
        Args:
            grid_state: Current grid state 
                       [batch_size, state_dim, grid_size, grid_size]
        
        Returns:
            action_logits: Logits for each possible backward action
        """
        state_embedding = self.encode_state(grid_state)
        action_logits = self.backward_net(state_embedding)
        return action_logits
    
    def compute_flow(self, grid_state):
        """
        Compute the flow (unnormalized) through a state
        
        Args:
            grid_state: Current grid state 
                       [batch_size, state_dim, grid_size, grid_size]
        
        Returns:
            flow: Flow value (unnormalized) for the state
        """
        state_embedding = self.encode_state(grid_state)
        flow = self.flow_net(state_embedding)
        # Apply softplus to ensure positive flow
        flow = F.softplus(flow)
        return flow
    
    def sample_trajectory(self, batch_size=1, max_length=None):
        """
        Sample trajectories from the GFlowNet
        
        Args:
            batch_size: Number of trajectories to sample
            max_length: Maximum trajectory length (defaults to grid_size*grid_size)
            
        Returns:
            states: List of states in the trajectory
            actions: List of actions taken
            terminated: Whether each trajectory reached termination
        """
        if max_length is None:
            max_length = self.grid_size * self.grid_size
            
        device = next(self.parameters()).device
        
        # Initialize states as empty grids
        current_states = torch.zeros(batch_size, self.state_dim, 
                                     self.grid_size, self.grid_size,
                                     device=device)
        
        # Channel 0: current grid
        # Channel 1: position markers for last action
        # Channel 2: visitation map
        
        # Start at center position
        pos_x, pos_y = self.grid_size // 2, self.grid_size // 2
        current_states[:, 1, pos_y, pos_x] = 1.0  # Mark initial position
        
        states = [current_states.clone()]
        actions = []
        terminated = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        # Sample trajectory
        for _ in range(max_length):
            # Get policy logits
            logits = self.forward_policy(current_states)
            
            # Mask invalid actions
            # (Here we would add logic to prevent invalid moves like going out of bounds)
            
            # Sample actions
            action_probs = F.softmax(logits, dim=-1)
            action_dist = torch.distributions.Categorical(action_probs)
            action = action_dist.sample()
            
            # Store action
            actions.append(action)
            
            # Apply actions and update states
            new_states = current_states.clone()
            
            # Process each batch element
            for b in range(batch_size):
                if terminated[b]:
                    continue
                    
                # Get current position
                y_pos, x_pos = torch.where(current_states[b, 1] > 0)
                if len(y_pos) > 0:  # If position is marked
                    y, x = y_pos[0].item(), x_pos[0].item()
                    
                    # Process action: 0,1,2,3 = up,right,down,left, 4 = stop
                    a = action[b].item()
                    
                    # Stop action
                    if a == 4:
                        terminated[b] = True
                        continue
                    
                    # Movement actions
                    new_y, new_x = y, x
                    if a == 0:  # Up
                        new_y = max(0, y - 1)
                    elif a == 1:  # Right
                        new_x = min(self.grid_size - 1, x + 1)
                    elif a == 2:  # Down
                        new_y = min(self.grid_size - 1, y + 1)
                    elif a == 3:  # Left
                        new_x = max(0, x - 1)
                    
                    # Update position marker
                    new_states[b, 1] = 0
                    new_states[b, 1, new_y, new_x] = 1
                    
                    # Mark path on grid and visitation map
                    new_states[b, 0, new_y, new_x] = 1  # Mark cell as visited
                    new_states[b, 2, new_y, new_x] += 1  # Increment visit count
            
            # Update states
            current_states = new_states
            states.append(current_states.clone())
            
            # Check if all trajectories are done
            if terminated.all():
                break
                
        return states, actions, terminated
    
    def compute_trajectory_flow(self, states, actions):
        """
        Compute flows for a trajectory
        
        Args:
            states: List of states in the trajectory
            actions: List of actions taken
            
        Returns:
            forward_flows: Forward transition flows
            backward_flows: Backward transition flows
            state_flows: State flows
        """
        forward_flows = []
        backward_flows = []
        state_flows = []
        
        # Compute flow for each state
        for state in states:
            state_flow = self.compute_flow(state)
            state_flows.append(state_flow)
        
        # Compute transition flows
        for i in range(len(states)-1):
            state = states[i]
            next_state = states[i+1]
            action = actions[i]
            
            # Forward flow
            forward_logits = self.forward_policy(state)
            forward_probs = F.softmax(forward_logits, dim=-1)
            forward_flow = state_flows[i] * forward_probs.gather(1, action.unsqueeze(1)).squeeze()
            forward_flows.append(forward_flow)
            
            # Backward flow
            if i > 0:  # Skip for first transition
                backward_logits = self.backward_policy(next_state)
                backward_probs = F.softmax(backward_logits, dim=-1)
                # Here we assume action indices for backward policy align with forward
                backward_flow = state_flows[i+1] * backward_probs.gather(1, action.unsqueeze(1)).squeeze()
                backward_flows.append(backward_flow)
        
        return forward_flows, backward_flows, state_flows
    
    def compute_trajectory_balance_loss(self, states, actions, rewards):
        """
        Compute flow matching or trajectory balance loss
        
        Args:
            states: List of states in the trajectory
            actions: List of actions taken
            rewards: Terminal rewards for the trajectories
            
        Returns:
            loss: Flow matching loss
        """
        forward_flows, backward_flows, state_flows = self.compute_trajectory_flow(states, actions)
        
        # Compute trajectory balance loss
        loss = 0
        
        # Terminal flow should match reward
        terminal_flow_error = F.mse_loss(state_flows[-1].squeeze(), rewards)
        loss += terminal_flow_error
        
        # Flow matching at each state
        for i in range(len(forward_flows)):
            if i < len(backward_flows):
                # Flow matching: in-flow should equal out-flow
                flow_match_error = F.mse_loss(forward_flows[i], backward_flows[i])
                loss += flow_match_error
        
        return loss
    