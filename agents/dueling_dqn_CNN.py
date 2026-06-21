import torch
import torch.nn as nn

class DuelingDQN(nn.Module):
    def __init__(self, state_dim=3087, action_dim=5, num_beams=1024, frame_stack=3):
        """
        State dim calculation: (1024 beams * 3 frames) + (5 kinematics * 3 frames) = 3087
        """
        super(DuelingDQN, self).__init__()
        
        self.num_beams = num_beams
        self.frame_stack = frame_stack
        self.kinematics_dim = 5 # [vx, vy, theta, omega, target_angle]

        # 1D CNN Spatial Extractor for sequential array parsing
        # Input shape expected: (Batch, Channels=frame_stack, Sequence_Length=256)
        self.lidar_extractor = nn.Sequential(
            nn.Conv1d(frame_stack, 32, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),

            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),

            nn.Conv1d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),

            nn.Flatten()
        )
        # Calculate CNN output feature mapping size dynamically
        with torch.no_grad():
            dummy_lidar = torch.zeros(1, frame_stack, num_beams)
            cnn_out_features = self.lidar_extractor(dummy_lidar).shape[1]
            
        # Total hidden feature size joining spatial structures and state parameters
        total_dense_input = cnn_out_features + (self.kinematics_dim * frame_stack)
        
        # Shared dense block
        self.shared_feature_layer = nn.Sequential(
            nn.Linear(total_dense_input, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )

        # Dueling Split: Value Stream V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        # Dueling Split: Advantage Stream A(s,a)
        self.advantage_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        batch_size = x.shape[0]
        
        x = x.view(batch_size, self.frame_stack, self.num_beams + self.kinematics_dim)

        lidar_spatial = x[:, :, :self.num_beams]          # (B, 3, 256)
        kinematics = x[:, :, self.num_beams:].reshape(batch_size, -1)

        # Extract features across both distinct data streams
        spatial_features = self.lidar_extractor(lidar_spatial)
        combined_features = torch.cat([spatial_features, kinematics], dim=1)
        
        shared_out = self.shared_feature_layer(combined_features)
        
        value = self.value_stream(shared_out)
        advantages = self.advantage_stream(shared_out)

        # Q(s,a) = V(s) + (A(s,a) - Mean(A(s,a)))
        return value + (advantages - advantages.mean(dim=1, keepdim=True))