from rnn import RNNCell
from train import RNNTrainer, TextDataset

# Initialize model and trainer
with open("./input.txt", "r") as f:
    data = f.read()

text_dataset = TextDataset(data, seq_length=16)

input_size = text_dataset.vocab_size
hidden_size = 100
output_size = text_dataset.vocab_size

rnn_cell = RNNCell(input_size, hidden_size, output_size)
trainer = RNNTrainer(
    rnn_cell,
    text_dataset,
    hidden_size,
    batch_size=1,
    learning_rate=0.1,
)

# Train
loss_log = trainer.train(max_data=1000000)
