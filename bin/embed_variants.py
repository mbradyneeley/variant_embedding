import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

# Load the CSV file
file_path = '../data/variant_interpretations.csv'
df = pd.read_csv(file_path)

# Extract the second column (sentence) as queries
queries = df['sentence'].tolist()

# Load model with tokenizer
model = SentenceTransformer('nvidia/NV-Embed-v2', trust_remote_code=True)
model.max_seq_length = 32768
model.tokenizer.padding_side = "right"

# Add EOS tokens to inputs
def add_eos(input_examples):
    return [input_example + model.tokenizer.eos_token for input_example in input_examples]

# Generate embeddings for queries (from the CSV file)
batch_size = 2
query_embeddings = model.encode(add_eos(queries), batch_size=batch_size, normalize_embeddings=True)

# Reduce dimensionality using PCA to visualize in 3D
pca = PCA(n_components=3)
reduced_embeddings = pca.fit_transform(query_embeddings)

# Matplotlib 3D plotting
fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(projection='3d')

# Plot each sample with a simple scatter
ax.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], reduced_embeddings[:, 2])

# Set axis labels
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_zlabel('Principal Component 3')

# Save the plot to a file in the data directory
output_file = '../data/variant_embeddings_plot.png'
plt.savefig(output_file)

# Show plot (optional)
plt.show()

# Print similarity scores between queries
scores = (query_embeddings @ query_embeddings.T) * 100
print("Similarity Scores:")
print(scores.tolist())

