import torch
import torch.nn as nn

class CustomJointNetwork(nn.Module):
    def __init__(self, joint_dim, num_outputs):
        super(CustomJointNetwork, self).__init__()
        self.tanh = nn.Tanh()
        self.linear = nn.Linear(joint_dim, num_outputs)

    def forward(self, enc_out, pred_out):

        out = enc_out + pred_out
        # out = F.relu(out)
        out = self.tanh(out)
        out = self.linear(out)
        return out

class JointNetwork(nn.Module):
    def __init__(
        self,
        enc_embed_dim,
        dec_embed_dim,
        joint_dim,
        num_outputs,
    ):
        super(JointNetwork, self).__init__()
        self.linear_enc = nn.Linear(enc_embed_dim, joint_dim)
        self.linear_dec = nn.Linear(dec_embed_dim, joint_dim)
        self.linear_out = nn.Linear(joint_dim, num_outputs)
        self.tanh = nn.Tanh()

    def forward(self, enc_out, pred_out, is_aux: bool=False):
    
        if enc_out.dim() == 3 and pred_out.dim() == 3:  # training
            seq_lens = enc_out.size(1)
            target_lens = pred_out.size(1)
            
            enc_out = enc_out.unsqueeze(2)
            pred_out = pred_out.unsqueeze(1)
            
            enc_out = enc_out.repeat(1, 1, target_lens, 1)
            pred_out = pred_out.repeat(1, seq_lens, 1, 1)
        else:
            assert enc_out.dim() == pred_out.dim()
        
        if is_aux:
            out = self.tanh(enc_out + self.linear_dec(pred_out))
        else:
            out = self.tanh(self.linear_enc(enc_out) + self.linear_dec(pred_out))
        
        out = self.linear_out(out)

        return out