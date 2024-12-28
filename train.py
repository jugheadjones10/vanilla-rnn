import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


class TextDataset(Dataset):
    def __init__(self, text_data: str, seq_length: int = 16) -> None:
        """Creates a dataset of character sequences from input text.

        Args:
            text_data: Full text data as a single string.
            seq_length: Length of character sequences per dataset index.
        """
        self.chars = sorted(list(set(text_data)))
        self.data_size, self.vocab_size = len(text_data), len(self.chars)
        # useful way to fetch characters either by index or char
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.seq_length = seq_length
        self.X = self.string_to_vector(text_data)

    @property
    def X_string(self) -> str:
        """Returns X in string form."""
        return self.vector_to_string(self.X)

    def __len__(self) -> int:
        """Returns the number of sequences in the dataset.

        Note:
            We remove the last sequence to avoid conflicts with Y being shifted right.
            This causes our model to never see the last sequence of text.

        Returns:
            int: Number of sequences in the dataset.
        """
        return int(len(self.X) / self.seq_length - 1)

    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor]:
        """Gets a sequence pair (X, Y) where Y is X shifted right by one position.

        Args:
            index: The sequence index to retrieve.

        Returns:
            tuple: Contains (X, Y) where:
                X is the input sequence tensor of shape (seq_length,)
                Y is the target sequence tensor of shape (seq_length,)
        """
        start_idx = index * self.seq_length
        end_idx = (index + 1) * self.seq_length

        X = torch.tensor(self.X[start_idx:end_idx]).float()
        y = torch.tensor(self.X[start_idx + 1 : end_idx + 1]).float()
        return X, y

    def string_to_vector(self, name: str) -> list[int]:
        """Converts a string into a vector of character indices.

        Args:
            name: Input string to convert.

        Returns:
            list[int]: List of integer indices representing the characters.

        Example:
            >>> string_to_vector('test')
            [20, 5, 19, 20]
        """
        vector = list()
        for s in name:
            vector.append(self.char_to_idx[s])
        return vector

    def vector_to_string(self, vector: list[int]) -> str:
        """Converts a vector of character indices back into a string.

        Args:
            vector: List of integer indices representing characters.

        Returns:
            str: The reconstructed string.

        Example:
            >>> vector_to_string([20, 5, 19, 20])
            'test'
        """
        vector_string = ""
        for i in vector:
            vector_string += self.idx_to_char[i]
        return vector_string


# Class for training RNN with Pytorch
class RNNTrainer:
    def __init__(
        self,
        rnn_cell,
        text_dataset,
        hidden_size,
        batch_size=64,
        learning_rate=0.1,
        seed=42,
        output_file="rnn_output.txt",  # Add output file for tracking
    ):
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.rnn_cell = rnn_cell
        self.vocab_size = text_dataset.vocab_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size

        # Data
        self.text_dataset = text_dataset
        self.dataloader = DataLoader(self.text_dataset, batch_size=batch_size)

        # Logging
        self.output_file = output_file
        with open(self.output_file, "w") as f:
            f.write("")
        self.loss_log = []

        # Training components
        self.optimizer = torch.optim.Adagrad(
            self.rnn_cell.parameters(), lr=learning_rate
        )
        self.loss_fn = nn.CrossEntropyLoss()

    def train(
        self,
        max_data=1000000,
        print_every=200,
        sample_every=1000,
    ):
        self.n = 0
        p = 0

        while p < max_data:
            for X, Y in self.dataloader:
                h_prev = self.rnn_cell.init_hidden(batch_size=X.shape[0])

                # Train step
                loss, h_prev = self.train_step(X, Y, h_prev)

                # Logging
                self.loss_log.append(loss)
                if self.n % print_every == 0:
                    print(f"Iteration {self.n}, Chars processed: {p}, Loss: {loss:.4f}")

                if self.n % sample_every == 0:
                    generated_text = self.sample(
                        h_prev, start_char=int(X[0, 0])
                    )  # Sample some text
                    print(f"----\n {generated_text} \n----")

                    # Append sample text to file
                    with open(self.output_file, "a") as f:
                        f.write(f"\nIteration {self.n}")
                        f.write(f"\nLoss: {loss:.4f}")
                        f.write("\n" + "=" * 50 + "\n")
                        f.write(generated_text)
                        f.write("\n" + "=" * 50 + "\n")

                # Update counters
                p += X.shape[0] * X.shape[1]  # batch_size * seq_length
                self.n += 1

                # Check if we've processed enough data
                if p >= max_data:
                    break

        return self.loss_log

    def train_step(self, X, Y, h_prev):
        """Performs a single training step for the RNN.

        Args:
            X: Input tensor of shape [batch_size, seq_length].
            Y: Target tensor of shape [batch_size, seq_length].
            h_prev: Previous hidden state tensor of shape [batch_size, hidden_size].
        """
        self.optimizer.zero_grad()
        loss = 0

        for t in range(X.shape[1]):  # iterate through sequence
            # Convert input to one-hot vectors [batch_size, vocab_size]
            x_t = F.one_hot(X[:, t].long(), num_classes=self.vocab_size).float()

            # Forward pass
            h_prev, y_t = self.rnn_cell(x_t, h_prev)

            # Compute loss for this time step
            loss_t = self.loss_fn(y_t, Y[:, t].long())
            loss += loss_t

        loss.backward()

        nn.utils.clip_grad_norm_(self.rnn_cell.parameters(), 5)

        self.optimizer.step()

        return loss.item(), h_prev

    def sample(self, h_prev, start_char, length=200):
        """Samples text from the RNN model.

        Args:
            h_prev: Previous hidden state tensor of shape [batch_size, hidden_size].
            start_char: Index of the starting character.
            length: Length of the generated text.

        Returns:
            str: Generated text.
        """
        x_t = torch.zeros(1, self.vocab_size)
        x_t[0, start_char] = 1
        output = [self.text_dataset.idx_to_char[start_char]]

        for _ in range(length):
            # Forward pass
            h_prev, y_t = self.rnn_cell(x_t, h_prev)

            result = torch.multinomial(
                nn.functional.softmax(y_t, dim=1), num_samples=1
            ).item()

            # Convert index to character and append
            char = self.text_dataset.idx_to_char[result]
            output.append(char)

            # Prepare next input
            x_t = torch.zeros(1, self.vocab_size)
            x_t[0, result] = 1

        # Save and print output
        generated_text = "".join(output)
        return generated_text
