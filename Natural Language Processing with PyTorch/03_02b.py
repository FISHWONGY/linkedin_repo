import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
  def __init__(self, vocabulary_size, embedding_size, 
               kernels_number, kernel_sizes, output_size, dropout_rate):


  def forward(self, text):
    

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
model.to(device)
