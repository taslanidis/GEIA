import torch
import torch.nn as nn

# Sentence Model
class Sentence_Embedding_model(nn.Module):
    def __init__(self, embedding_dim: int = 718, sequence_length:int = 10, model_type:str ="mean"):
        super(Sentence_Embedding_model, self).__init__()

        if model_type == "mean":
            self.model = MeanMapping()
        elif model_type == "linear":
            self.model = LinearMapping(sequence_length)
        elif model_type == "convolution":
            self.model = Conv1DMapping(sequence_length)
        elif model_type == "self-attention":
            self.model = MultiheadAttentionMapping(embedding_dim, num_heads=4)
        elif model_type == "encoder":
            raise Exception("Not implemented")
        else:
            raise Exception("Not understand")
    
    def forward(self,x):
        return self.model(x) 

class MeanMapping(nn.Module):
    def __init__(self):
        super(MeanMapping, self).__init__()

    def forward(self, x):
        return torch.mean(x,dim=1)

class LinearMapping(nn.Module):
    def __init__(self, sequence_length):
        super(LinearMapping, self).__init__()
        self.fc1 = nn.Linear(sequence_length, 1)

    def forward(self, x):
        batch,length,embedding = x.shape
        x_permuted = x.permute(0,2,1)
        out = self.fc1(x_permuted)
        out = out.squeeze()
        out = out.reshape(batch,embedding)
        return out

class Conv1DMapping(nn.Module):
    def __init__(self, sequence_length):
        super(Conv1DMapping, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=sequence_length, 
                               out_channels=sequence_length // 2, 
                               kernel_size=3, 
                               padding=1)  # Reduce sequence length by half
        self.conv2 = nn.Conv1d(in_channels=sequence_length // 2, 
                               out_channels=1, 
                               kernel_size=3, 
                               padding=1)  # Collapse to 1 channel

    def forward(self, x):
        # x is of shape [batch, length, embedding]
        batch,length,embedding = x.shape
        x = self.conv1(x)  # Shape: [batch, length / 2, embedding]
        x = torch.relu(x)
        x = self.conv2(x)  # Shape: [batch, 1, embedding]
        x = x.reshape(batch,embedding)  # Shape: [batch, embedding]
        return x

class MultiheadAttentionMapping(nn.Module):
    def __init__(self, embedding_dim, num_heads):
        super(MultiheadAttentionMapping, self).__init__()
        # MultiheadAttention layer
        self.multihead_attention = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=num_heads, batch_first=True)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, sequence_length, embedding_dim)
        Returns:
            sentence_embedding: Tensor of shape (batch_size, embedding_dim)
        """
        # Use the input as both query, key, and value for self-attention
        # self.multihead_attention expects (batch_size, sequence_length, embedding_dim)
        attended_output, _ = self.multihead_attention(x, x, x)
        
        # Aggregate sequence features to create a sentence embedding
        # Here, using mean pooling. Other methods (e.g., max pooling, weighted sum) can also be used.
        sentence_embedding = attended_output.mean(dim=1)  # (batch_size, embedding_dim)

        return sentence_embedding