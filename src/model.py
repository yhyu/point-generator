import numpy as np
import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_


def get_angles(pos, i, d):
    return pos/np.power(10000, 2*(i//2)/d)

def create_look_ahead_mask(size):
    return torch.triu(torch.ones(size, size), diagonal=1)

def positional_encoding(positions, d, device):
    angle_rads = get_angles(np.arange(positions)[:, np.newaxis],
                            np.arange(d)[np.newaxis, :],
                            d)
  
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    pos_encoding = angle_rads[np.newaxis, ...]
    
    return torch.FloatTensor(pos_encoding).to(device)


def FullyConnected(embedding_dim, fully_connected_dim, device):
    model = nn.Sequential(
        nn.Linear(embedding_dim, fully_connected_dim),
        nn.ReLU(),
        nn.Linear(fully_connected_dim, embedding_dim)
    ).to(device)
    return model


class DecoderLayer(nn.Module):
    def __init__(self, embedding_dim, num_heads, fully_connected_dim, device, dropout_rate=0.1, layernorm_eps=1e-6):
        super(DecoderLayer, self).__init__()
        
        self.mha1 = nn.MultiheadAttention(embedding_dim, num_heads, dropout=dropout_rate,
                                          batch_first=True, device=device)
        self.mha2 = nn.MultiheadAttention(embedding_dim, num_heads, dropout=dropout_rate,
                                          batch_first=True, device=device)
        self.ffn = FullyConnected(embedding_dim, fully_connected_dim, device=device)
        
        self.layernorm1 = nn.LayerNorm(fully_connected_dim, eps=layernorm_eps, device=device)
        self.layernorm2 = nn.LayerNorm(fully_connected_dim, eps=layernorm_eps, device=device)
        self.layernorm3 = nn.LayerNorm(fully_connected_dim, eps=layernorm_eps, device=device)
        
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.dropout3 = nn.Dropout(dropout_rate)
    
    def forward(self, x, enc_output, dec_attention_mask, dec_padding_mask, enc_padding_mask):
        attn1, _ = self.mha1(x, x, x, key_padding_mask=dec_padding_mask, attn_mask=dec_attention_mask)
        attn1 = self.dropout1(attn1)
        out1 = self.layernorm1(attn1 + x)
        
        attn2, _ = self.mha2(out1, enc_output, enc_output, key_padding_mask=enc_padding_mask)
        attn2 = self.dropout2(attn2)
        out2 = self.layernorm2(attn2 + out1)
        
        ffn_output = self.ffn(out2)
        ffn_output = self.dropout2(ffn_output)
        out3 = self.layernorm3(ffn_output + out2)
        return out3


class Decoder(nn.Module):
    def __init__(self, num_layers, embedding_dim, num_heads, fully_connected_dim, num_items_pos, num_items_color,
                 maximum_position_encoding, dropout_rate=0.1, layernorm_eps=1e-6, device='cpu'):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        
        self.embedding_pos = nn.Embedding(
            num_items_pos+2, self.embedding_dim, padding_idx=num_items_pos+1, device=device)
        self.embedding_color = nn.Embedding(
            num_items_color+1, self.embedding_dim, padding_idx=num_items_color, device=device)
        self.pos_encoding = positional_encoding(maximum_position_encoding, self.embedding_dim, device)
        
        self.dec_layers = nn.ParameterList(
            [DecoderLayer(embedding_dim=self.embedding_dim,
                          num_heads=num_heads,
                          fully_connected_dim=fully_connected_dim,
                          device=device,
                          dropout_rate=dropout_rate,
                          layernorm_eps=layernorm_eps)
                for _ in range(self.num_layers)])
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x, enc_output, dec_attention_mask, dec_padding_mask, enc_padding_mask):
        seq_len = x.shape[1]
        
        x = torch.concat(
            [
                self.embedding_pos(x[:, :3]),
                self.embedding_color(x[:, 3:])
            ],
            axis=1
        )
        x *= torch.sqrt(torch.tensor(self.embedding_dim))
        x += self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x)
        
        for i in range(self.num_layers):
            x = self.dec_layers[i](
                x, enc_output, dec_attention_mask, dec_padding_mask, enc_padding_mask)
        
        return x


class Generator(nn.Module):
    def __init__(self, num_items_pos: int, num_items_color: int, num_items_out: list,
                 num_layers: int = 6, num_heads: int = 16,
                 embedding_dim: int = 256, hidden_size: int = 256,
                 dropout_rate: float = 0.1,
                 max_item_list_length: int = 5,
                 device: str = 'cpu') -> None:
        super().__init__()

        self.embedding_dim = embedding_dim
        self.BOS = num_items_pos

        self.decoder = Decoder(
            num_layers, embedding_dim=embedding_dim,
            num_items_pos=num_items_pos, num_items_color=num_items_color,
            num_heads=num_heads, fully_connected_dim=hidden_size,
            maximum_position_encoding=max_item_list_length,
            dropout_rate=dropout_rate, device=device,
        )

        self.classifier = nn.ParameterList(
            [
                nn.Linear(embedding_dim, num)
                for num in num_items_out
            ]
        )

        # parameters initialization
        self.apply(self._init_weights)
    
    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Embedding) or isinstance(module, nn.Linear):
            xavier_normal_(module.weight)

    def forward(self, inp, noise, enc_padding_mask, dec_padding_mask, dec_attention_mask, **kwargs):
        
        # decoder
        dec_output = self.decoder(
            inp, noise, dec_attention_mask, dec_padding_mask, enc_padding_mask)
        
        # classifier
        final_output = [
            cls(dec_output[:, i]) for i, cls in enumerate(self.classifier[:inp.shape[1]])
        ]
        
        return final_output
    
    def generate(self, num: int):
        self.eval()

        noises = torch.randn(num, 1, self.embedding_dim)
        outputs = torch.tensor([], dtype=torch.long)
        for i in range(5):
            inputs = torch.cat([torch.tensor([self.BOS]*noises.shape[0]).view(-1, 1), outputs], axis=1)
            with torch.no_grad():
                logits = self.forward(inputs, noises, None, None, None)
            predictions = torch.argmax(logits[-1], dim=-1)
            outputs = torch.cat([outputs, predictions[:].view(-1, 1)], dim=-1)
        return outputs
