import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any, Tuple
import logging
from datetime import datetime

class TransformerClassifier(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Transformer-based classifier
        
        Args:
            config: Configuration dictionary containing:
                - input_dim: Input feature dimension
                - num_heads: Number of attention heads
                - num_layers: Number of transformer layers
                - dropout: Dropout rate
        """
        super().__init__()
        self.config = config
        self._build_model()

    def _build_model(self):
        """Build transformer model architecture"""
        self.input_dim = self.config['input_dim']
        self.num_heads = self.config['num_heads']
        self.num_layers = self.config['num_layers']
        
        # Input embedding
        self.input_embedding = nn.Linear(self.input_dim, self.config['hidden_dim'])
        
        # Transformer layers
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=self.config['hidden_dim'],
            nhead=self.num_heads,
            dim_feedforward=self.config['ff_dim'],
            dropout=self.config['dropout']
        )
        self.transformer = nn.TransformerEncoder(
            transformer_layer, 
            num_layers=self.num_layers
        )
        
        # Output layers
        self.classifier = nn.Linear(self.config['hidden_dim'], 
                                  self.config['num_classes'])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass"""
        # Input embedding
        x = self.input_embedding(x)
        
        # Transformer encoding
        attention_maps = []
        for layer in self.transformer.layers:
            # Self-attention
            attended, attention = layer.self_attn(x, x, x, 
                                                need_weights=True)
            attention_maps.append(attention)
            
            # Add & Norm
            x = layer.norm1(x + layer.dropout1(attended))
            
            # Feed-forward
            ff = layer.linear2(F.relu(layer.linear1(x)))
            x = layer.norm2(x + layer.dropout2(ff))
        
        # Classification
        logits = self.classifier(x)
        
        return logits, torch.stack(attention_maps)

class DeepLearningClassifier:
    def __init__(self, config: Dict[str, Any]):
        """Initialize classifier with configuration"""
        self.config = config
        self.logger = logging.getLogger(__name__)
        self._initialize_components()

    def _initialize_components(self):
        """Initialize classifier components"""
        # Initialize model
        self.model = TransformerClassifier(self.config)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.get('learning_rate', 1e-4)
        )

    def classify_container(self, features: torch.Tensor) -> Dict[str, Any]:
        """
        Classify container based on features
        
        Args:
            features: Tensor of container features
            
        Returns:
            Dictionary containing classification results and attention maps
        """
        try:
            self.model.eval()
            with torch.no_grad():
                # Forward pass
                logits, attention_maps = self.model(features)
                
                # Get predictions
                probabilities = F.softmax(logits, dim=-1)
                predictions = torch.argmax(probabilities, dim=-1)
                
                # Compute confidence
                confidence = torch.max(probabilities, dim=-1)[0]
                
                return {
                    'predictions': predictions.cpu().numpy(),
                    'probabilities': probabilities.cpu().numpy(),
                    'confidence': confidence.cpu().numpy(),
                    'attention_maps': attention_maps.cpu().numpy()
                }
                
        except Exception as e:
            self.logger.error(f"Classification failed: {str(e)}")
            raise

    def update_model(self, 
                    features: torch.Tensor, 
                    labels: torch.Tensor) -> Dict[str, float]:
        """Update model with new data"""
        try:
            self.model.train()
            
            # Forward pass
            logits, _ = self.model(features)
            
            # Compute loss
            loss = F.cross_entropy(logits, labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Compute metrics
            accuracy = (torch.argmax(logits, dim=-1) == labels).float().mean()
            
            return {
                'loss': loss.item(),
                'accuracy': accuracy.item()
            }
            
        except Exception as e:
            self.logger.error(f"Model update failed: {str(e)}")
            raise
