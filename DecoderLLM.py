#%%
import torch.nn as nn
import torch
import torch.nn.functional as F
import pandas as pd
import re
import contractions
import json
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
import torch.optim as optim
from tensorflow.keras.preprocessing.sequence import pad_sequences
from torch.cuda.amp import GradScaler, autocast
#%%

# Preprocessing Functions
def process_dialogues(file_path):
    dialogues = pd.read_csv(file_path, delimiter='\t', names=['Prompt', 'Response'])
    return dialogues

def expand_contractions(text):
    return contractions.fix(text)

def tokenize_text(text):
    text = expand_contractions(text.lower())
    text = re.sub(r'([.,?!])', r' \1 ', text)  # Space out punctuation
    text = re.sub(r'\s+', ' ', text).strip()
    return text.split()

def tokenize_dataframe(df):
    df['Prompt'] = df['Prompt'].apply(tokenize_text)
    df['Response'] = df['Response'].apply(tokenize_text)
    return df

  # Example usage:
file_path = '/content/dialogs.txt'  # Update this if needed
df = process_dialogues(file_path)
df = tokenize_dataframe(df)
print(df.head())
#%%
# Vocabulary Builder
def build_vocabulary(df, maxlen=None):
    vocab = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
    index = 4
    for column in ['Prompt', 'Response']:
        for sample in df[column]:
            for word in sample:
                if word not in vocab:
                    vocab[word] = index
                    index += 1
                    if maxlen and len(vocab) >= maxlen:
                        return vocab
    return vocab

# Example usage:
# Assuming df is the processed dataset where each sample is a list of words
# df = tokenize_dataframe(df)  # Ensure the dataset is tokenized before using this
vocab = build_vocabulary(df, maxlen=5000)  # Set a max length if needed
print(vocab)
#%%
def char_to_int(vocab, char):
    """
    Convert a character (word) to an integer based on the vocabulary mapping.
    If the character is not in the vocabulary, return the <UNK> token index.
    """
    return vocab.get(char, vocab["<UNK>"])

def int_to_char(vocab, index):
    """
    Convert an integer index to a character (word) based on the vocabulary mapping.
    If the index is out of range, return <UNK> token.
    """
    # Reverse the vocab dictionary (int -> word)
    inv_vocab = {v: k for k, v in vocab.items()}
    return inv_vocab.get(index, "<UNK>")

# Example Usage:
# Assuming vocab is the vocabulary dictionary as built from your code

vocab = build_vocabulary(df)  # Assuming df is the DataFrame containing your data

# Convert character to integer
char = "<UNK>"
char_index = char_to_int(vocab, char)
print(f"Character '{char}' is converted to integer: {char_index}")

# Convert integer to character
index = 3
index_char = int_to_char(vocab, index)
print(f"Integer {index} is converted to character: {index_char}")
#%%
labeledDF=pd.DataFrame()
labeledDF['prompt_int'] = df['Prompt'].apply(lambda x: [char_to_int(vocab, word) for word in x])
labeledDF['response_int'] = df['Response'].apply(lambda x: [char_to_int(vocab, word) for word in x])
labeledDF.head()
#%%
# Find the length of the longest list in 'prompt_int' and 'reponse_int'
max_prompt_length = labeledDF['prompt_int'].apply(len).max()
max_response_length = labeledDF['response_int'].apply(len).max()
longer_length = max(max_prompt_length, max_response_length) if max_prompt_length > max_response_length else max_response_length
# Print the results
print(f"Longest prompt length: {max_prompt_length}")
print(f"Longest response length: {max_response_length}")
#%%
labeledDF.head()
#%%


# Define the maxlen for padding
maxlen = longer_length  # Set to desired max length

# Function to pad sequences
def pad_data(sequences, maxlen):
    return pad_sequences(sequences, maxlen=maxlen, padding='post', truncating='post', value=0)

# Apply padding to both 'prompt_int' and 'response_int' columns
labeledDF['prompt_int'] = labeledDF['prompt_int'].apply(lambda x: pad_data([x], maxlen)[0])
labeledDF['response_int'] = labeledDF['response_int'].apply(lambda x: pad_data([x], maxlen)[0])

#%%
labeledDF.head()
#%%

# Custom Dataset
class DialogueDataset(Dataset):
    def __init__(self, dataframe):
        self.data = dataframe

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        prompt = torch.tensor(self.data.iloc[idx]['prompt_int'], dtype=torch.long)
        response = torch.tensor(self.data.iloc[idx]['response_int'], dtype=torch.long)
        return prompt, response

# Create Dataset and DataLoader
dataset = DialogueDataset(labeledDF)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Testing the DataLoader
for batch_idx, (prompt, response) in enumerate(dataloader):
    print(f"Batch {batch_idx + 1}:")
    print(prompt.shape)
    print(response.shape)
    print("Prompt:", prompt)
    print("Response:", response)
    break  # Just to test the first batch
#%%
class MaskedMultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, head_num):
        super().__init__()
        assert embed_dim % head_num == 0, "embed_dim must be divisible by head_num"
        self.head_num = head_num
        self.head_dim = embed_dim // head_num

        # Linear layers without bias
        self.q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v = nn.Linear(embed_dim, embed_dim, bias=False)
        self.projection = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        batch_size, seq_len, embed_dim = x.shape
        Q = self.q(x)
        K = self.k(x)
        V = self.v(x)

        # Reshape and transpose to prepare for multi-head attention
        Q = Q.view(batch_size, seq_len, self.head_num, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.head_num, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.head_num, self.head_dim).transpose(1, 2)

        # Calculate attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)

        # Create a lower triangular mask for causal attention
        mask = torch.tril(torch.ones(seq_len, seq_len)).to(x.device).unsqueeze(0).unsqueeze(0)
        scores = scores.masked_fill(mask == 0, float('-inf'))

        # Apply softmax to get attention weights
        attn = F.softmax(scores, dim=-1)

        # Compute the output of attention
        out = torch.matmul(attn, V)

        # Reshape and project the result back to embed_dim
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        return self.projection(out)

#%%
class FeedForward(nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim)
        )

    def forward(self, x):
        return self.layers(x)

#%%
# Decoder Block
class DecoderOnly(nn.Module):
    def __init__(self, embed_dim, head_num, hidden_dim):
        super().__init__()
        self.attention = MaskedMultiHeadAttention(embed_dim, head_num)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        attn_out = self.attention(self.norm1(x)) + x  # Residual Connection
        ff_out = self.ff(self.norm2(attn_out)) + attn_out  # Residual Connection
        return ff_out


#%%
# Define model hyperparameters
embed_dim = 512
head_num = 8
hidden_dim = 512
num_encoder = 6
num_decoder = 6
vocab_size=len(vocab)
#%%
# Define the positional encoding class
class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=5000):
        super().__init__()

        # Create a matrix of shape (max_len, embed_dim)
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * -(math.log(10000.0) / embed_dim))

        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # Add batch dimension
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: shape (batch_size, seq_len, embed_dim)
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]

# LLM Decoder Model with Positional Encoding
class DecoderLLM(nn.Module):
    def __init__(self, embed_dim, head_num, hidden_dim, vocab_size, num_decoder, max_len=5000):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)  # Use nn.Embedding instead of nn.Linear
        self.positional_encoding = PositionalEncoding(embed_dim, max_len)
        self.decoders = nn.ModuleList([DecoderOnly(embed_dim, head_num, hidden_dim) for _ in range(num_decoder)])
        self.norm = nn.LayerNorm(embed_dim)
        self.fc = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)  # x: shape (batch_size, seq_len) -> (batch_size, seq_len, embed_dim)
        x = self.positional_encoding(x)  # Add positional encoding
        for decoder in self.decoders:
            x = decoder(x)
        x = self.norm(x)
        return self.fc(x)  # No Softmax!

# Example usage
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DecoderLLM(embed_dim=embed_dim, head_num=head_num, hidden_dim=hidden_dim, vocab_size=vocab_size, num_decoder=num_decoder).to(device)
#%%
#TEST
# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model.eval()

# Iterate over the dataloader
for batch_idx, (data, target) in enumerate(dataloader):
    # Move data and target to the correct device
    data, target = data.to(device), target.to(device)

    # Forward pass through the model
    with torch.no_grad():
        output = model(data)  # Get the model's output
        last_token_logits = output[:, -1, :]
        probs = F.softmax(last_token_logits, dim=-1)
        predicted_next_word = torch.argmax(probs, dim=-1)
        predicted_next_words = [int_to_char(vocab, idx.item()) for idx in predicted_next_word]
        print(predicted_next_words)

    # Print or log the output for testing
    print(f"Batch {batch_idx + 1}:")
    #print("Output shape:", output.shape)
    #print("Output (first 5 values):", output[:5])  # Print the first 5 outputs
#%%
def train_with_teacher_forcing(model, dataloader, vocab, optimizer, device, epochs=10, teacher_forcing_ratio=0.5):
    """
    Train the model using cross-entropy loss with teacher forcing and mixed precision.
    """
    model.train()
    criterion = nn.CrossEntropyLoss()

    # Initialize GradScaler for mixed precision
    scaler = GradScaler()

    for epoch in range(epochs):
        epoch_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        for batch_idx, (data, target) in enumerate(dataloader):
            # Move data and target to the correct device
            data, target = data.to(device), target.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Initialize the input to be the start token or the data
            input_tensor = data

            # Set teacher forcing during training
            batch_size, seq_len = target.size()
            loss = 0

            # Loop over the sequence length (for decoder)
            for t in range(seq_len):
                # Forward pass through the model (use autocast for mixed precision)
                with autocast():  # Enable mixed precision
                    output = model(input_tensor)  # Assuming model returns (output, hidden_state)
                    output = output[:, -1, :]  # We want the last output (logits for the last time step)

                    # Use teacher forcing or model's own predictions
                    if torch.rand(1).item() < teacher_forcing_ratio:
                        # Teacher forcing: use the true token as the next input
                        input_tensor = target[:, t].unsqueeze(1)  # Get the true token for the next step
                    else:
                        # No teacher forcing: use the model's predicted token as the next input
                        predicted_token = output.argmax(dim=-1).unsqueeze(1)  # Get the predicted token
                        input_tensor = predicted_token  # Use the prediction as input for the next time step

                    # Compute loss at each time step
                    target_token = target[:, t]  # The actual token at this time step
                    loss += criterion(output, target_token)

            # Backpropagate the loss with mixed precision
            scaler.scale(loss).backward()

            # Optionally, clip gradients to prevent exploding gradients
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Update model parameters with mixed precision
            scaler.step(optimizer)
            scaler.update()  # Update the scaler for the next iteration

            epoch_loss += loss.item()

            # Compute accuracy (optional)
            _, predicted = output.max(1)  # Get the predicted word index
            correct_predictions += (predicted == target_token).sum().item()
            total_predictions += target_token.size(0)

        # Print statistics for this epoch
        epoch_accuracy = 100 * correct_predictions / total_predictions
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss / len(dataloader):.4f}, Accuracy: {epoch_accuracy:.2f}%")

# Example usage:
# Assume you have a model, dataloader, optimizer, and vocab
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
optimizer = optim.Adam(model.parameters(), lr=1e-3)

train_with_teacher_forcing(model, dataloader, vocab, optimizer, device, epochs=10)
#%%
def expand_contractions(text):
    return contractions.fix(text)

def tokenize_text(text):
    text = expand_contractions(text.lower())
    text = re.sub(r'([.,?!])', r' \1 ', text)  # Space out punctuation
    text = re.sub(r'\s+', ' ', text).strip()
    return text.split()

# Assuming char_to_int and int_to_char mappings exist
def char_to_int(vocab, text):
    return [vocab.get(word, vocab["<UNK>"]) for word in text]  # Convert words to indices

def int_to_char(token, vocab):
    return [k for k, v in vocab.items() if v == token][0]  # Convert indices to words

# Testing the model on a new text
text = "insert text here"
model.eval()

with torch.no_grad():
    maxlen = 25
    output_text = []
    text = tokenize_text(text)  # Tokenize input text
    text = torch.tensor(char_to_int(vocab, text), dtype=torch.long).unsqueeze(0).to(device)  # Convert to tensor

    for i in range(maxlen):
        output = model(text)  # Forward pass
        output = F.softmax(output[:, -1, :], dim=-1)  # Apply softmax on last token's prediction
        token = torch.argmax(output, dim=-1).item()  # Get predicted token

        if token == 2:  # Assuming <EOS> token index is 2
            break

        word = int_to_char(token, vocab)  # Convert back to word
        output_text.append(word)

        text = torch.cat([text, torch.tensor([[token]], dtype=torch.long).to(device)], dim=1)  # Append predicted token

    print(" ".join(output_text))  # Final generated output