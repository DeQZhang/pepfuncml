"""Generate synthetic peptide sequences from a trained recurrent generator."""

import torch
import numpy as np

# Define the amino acid alphabet with the common amino acids
amino_acids = 'ACDEFGHIKLMNPQRSTVWY'

# Define the model parameters
latent_dim = 10
hidden_dim = 128
num_classes = len(amino_acids)  # Number of amino acid classes

# Generator model based on LSTM.
class GeneratorRNN(torch.nn.Module):
    """Map latent noise vectors to amino-acid class probabilities over sequence positions."""
    def __init__(self, latent_dim, hidden_dim, num_classes):
        super(GeneratorRNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = torch.nn.LSTM(latent_dim, hidden_dim, batch_first=True)
        self.fc = torch.nn.Linear(hidden_dim, num_classes)

    def forward(self, z):
        """Run the recurrent generator and return per-position amino-acid probabilities."""
        h_0 = torch.zeros(1, z.size(0), self.hidden_dim).to(z.device)  # Initial hidden state
        c_0 = torch.zeros(1, z.size(0), self.hidden_dim).to(z.device)  # Initial memory cell
        out, _ = self.lstm(z, (h_0, c_0))
        out = self.fc(out)
        out = torch.softmax(out, dim=2)
        return out

def load_generator_model(model_path):
    """Load generator weights from disk and switch the model to inference mode."""
    generator = GeneratorRNN(latent_dim, hidden_dim, num_classes)
    generator.load_state_dict(torch.load(model_path))
    generator.eval()  # Switch to evaluation mode
    return generator

def generate_peptide(generator, num_sequences=1, sequence_length=10):
    """Sample latent noise and decode it into synthetic peptide logits."""
    noise = torch.randn(num_sequences, sequence_length, latent_dim).to(device)
    generated_sequences = generator(noise).detach().cpu().numpy()  # Move generated sequences from GPU to CPU
    return generated_sequences

def one_hot_to_sequence(one_hot_encoded):
    """Convert the model output at each sequence position into amino-acid letters."""
    indices = np.argmax(one_hot_encoded, axis=-1)
    sequence = ''.join([amino_acids[index] for index in indices])
    return sequence

if __name__ == "__main__":
    """Run a simple demo that generates several batches of synthetic peptides."""
    # Check whether a GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the generator model
    generator = load_generator_model('generator_model.pth')
    generator.to(device)

    # Generate peptide sequences
    num_sequences = 5       # Number of sequences to generate
    sequence_length = 60     # Length of each sequence

    for i in range(3):  # Generate 3 batches using different random noise
        print(f"Batch {i + 1}:")
        generated_sequences = generate_peptide(generator, num_sequences, sequence_length)
        decoded_sequences = [one_hot_to_sequence(seq) for seq in generated_sequences]

        # Output the generated peptide sequences
        for j, seq in enumerate(decoded_sequences, 1):
            print(f"Peptide {j}: {seq}")
        print("\n")
