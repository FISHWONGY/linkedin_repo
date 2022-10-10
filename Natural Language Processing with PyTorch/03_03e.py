import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
  def __init__(self, vocabulary_size, embedding_size,
                 kernels_number, kernel_sizes, output_size, dropout_rate):
    super().__init__()
    self.embedding = nn.Embedding(vocabulary_size, embedding_size)
    self.convolution_layers = nn.ModuleList([nn.Conv2d(in_channels=1, out_channels=kernels_number,
                                                         kernel_size=(k, embedding_size))
                                              for k in kernel_sizes])
    self.dropout = nn.Dropout(dropout_rate)
    self.fully_connected = nn.Linear(len(kernel_sizes) * kernels_number, output_size)

  def forward(self, text):
    text = text.permute(1, 0)
    input_embeddings = self.embedding(text)
    input_embeddings = input_embeddings.unsqueeze(1)
    conved = [F.relu(convolution_layer(input_embeddings)).squeeze(3) for convolution_layer in self.convolution_layers]
    pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
    concat = self.dropout(torch.cat(pooled, dim=1))
    final_output = self.fully_connected(concat)
    return final_output


vocabulary_size = 2679
embedding_size = 100
kernels_number = 100
kernel_sizes = [2, 3, 4]
output_size = 6
dropout_rate = 0.3

model = CNN(vocabulary_size, embedding_size, kernels_number, kernel_sizes, output_size, dropout_rate)

print(model)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
criterion = nn.CrossEntropyLoss()
criterion = criterion.to(device)


optimizer = torch.optim.Adam(model.parameters())


def accuracy(predictions, actual_label):
    max_predictions = predictions.argmax(dim=1, keepdim=True, )
    correct_predictions = max_predictions.squeeze(1).eq(actual_label)
    accuracy = correct_predictions.sum() / torch.cuda.FloatTensor([actual_label.shape[0]])
    return accuracy


def train(model, iterator, optimizer, criterion):

    model.train()
    epoch_loss = 0
    epoch_acc = 0
    
    for batch in iterator:
        optimizer.zero_grad()
        
        predictions = model(batch.text)
        
        loss = criterion(predictions, batch.label)
        
        acc = accuracy(predictions, batch.label)
        
        loss.backward()
        
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator, criterion):

    model.eval()
    epoch_loss = 0
    epoch_acc = 0
    
    with torch.no_grad():
    
        for batch in iterator:

            predictions = model(batch.text)
            
            loss = criterion(predictions, batch.label)
            
            acc = accuracy(predictions, batch.label)
            epoch_loss += loss.item()
            epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

