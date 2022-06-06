import torch
import torch.nn as nn

class TranscriptionnNetwork(nn.Module):
    def __init__(self, input_size, enc_dim, n_layers, bidirectional, dropout, output_dim):
        super(TranscriptionnNetwork, self).__init__()
        # self.embed = nn.Embedding(input_size, enc_dim)
        self.rnn = nn.GRU(input_size=input_size, hidden_size=enc_dim, num_layers=n_layers, batch_first=True, bidirectional=bidirectional, dropout=dropout)
        
        if bidirectional is True:
            self.linear = nn.Linear(enc_dim * 2, output_dim)
        else:
            self.linear = nn.Linear(enc_dim, output_dim)

    def forward(self, x):
        self.rnn.flatten_parameters()
        # out = self.embed(x) 
        # x = x.permute(0, 2, 1)
        out = self.rnn(x)[0]
        out = self.linear(out)
        return out