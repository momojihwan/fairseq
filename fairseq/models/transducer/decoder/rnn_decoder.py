import torch
import torch.nn as nn

class PredictionNetwork(nn.Module):
    def __init__(self, input_size, dec_dim, output_dim):
        super(PredictionNetwork, self).__init__()
        self.embed = nn.Embedding(input_size, dec_dim)
        self.rnn = nn.GRUCell(input_size=dec_dim, hidden_size=dec_dim)
        self.linear = nn.Linear(dec_dim, output_dim)

        self.initial_state = nn.Parameter(torch.randn(dec_dim))
        self.start_symbol = 0

    def forward_one_step(self, input, prev_state):
        embedding = self.embed(input)
        state = self.rnn.forward(embedding, prev_state)
        out = self.linear(state)
        return out, state

    def forward(self, y):
        batch_size = y.shape[0]
        U = y.shape[1]
        outs = []
        state = torch.stack([self.initial_state] * batch_size).to(y.device)
        for u in range(U+1):
            if u == 0:
                decoder_input = torch.tensor([self.start_symbol] * batch_size).to(y.device)
            else:
                decoder_input = y[:,u-1]
                
            out, state = self.forward_one_step(decoder_input, state)
            outs.append(out)
        out = torch.stack(outs, dim=1)
        return out