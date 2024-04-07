import torch
from torch.nn import MultiheadAttention

# Example parameters
embed_size = 768  # Embedding size
num_heads = 12     # Number of attention heads
seq_length = 7   # Sequence length
batch_size = 3    # Batch size

# Create the multi-head attention layer
mha_pytorch = MultiheadAttention(embed_dim=embed_size, num_heads=num_heads)


# Dummy input (batch_size, seq_length, embed_size)
input_tensor = torch.rand(batch_size, seq_length, embed_size)

# PyTorch expects inputs in the form (seq_length, batch_size, embed_size)
input_tensor = input_tensor.permute(1, 0, 2)

# Forward pass through the multi-head attention layer
# PyTorch MHA returns the output and attention weights
output_tensor, attn_weights = mha_pytorch(input_tensor, input_tensor, input_tensor)

print("PyTorch Output Shape:", output_tensor.shape)
for param in mha_pytorch.parameters():
    print(param.shape)
# print("PyTorch Attention Weights Shape:", attn_weights.shape)

import tensorflow as tf
from tensorflow.keras.layers import MultiHeadAttention

# Example parameters are the same as the PyTorch example

embed_size_keras = embed_size // num_heads  # Each head must have the same size

# Create the multi-head attention layer
mha_keras = MultiHeadAttention(num_heads=num_heads, key_dim=embed_size_keras)

# Dummy input (batch_size, seq_length, embed_size)
input_tensor = tf.random.uniform((batch_size, seq_length, embed_size))

# Forward pass through the multi-head attention layer
# Keras MHA returns only the output, not the attention weights directly
output_tensor = mha_keras(input_tensor, input_tensor)

print("Keras Output Shape:", output_tensor.shape)

# Access the weights
weights = mha_keras.get_weights()

# Print information about the weights
print(f"Total Weight Matrices: {len(weights)}")
for i, weight_matrix in enumerate(weights):
    print(f"Weight Matrix {i}: Shape {weight_matrix.shape}")
