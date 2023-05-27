import numpy as np
import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_
from tqdm.auto import tqdm


def get_angles(pos, i, d):
    return pos/np.power(10000, 2*(i//2)/d)

def create_look_ahead_mask(size):
    return torch.triu(torch.ones(size, size), diagonal=1) > 0

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
    def __init__(self, embedding_dim, num_heads, fully_connected_dim, device,
                 cross_attend=True, dropout_rate=0.1, layernorm_eps=1e-6):
        super(DecoderLayer, self).__init__()

        self.mha1 = nn.MultiheadAttention(embedding_dim, num_heads, dropout=dropout_rate,
                                          batch_first=True, device=device)
        self.ffn = FullyConnected(embedding_dim, fully_connected_dim, device=device)

        self.layernorm1 = nn.LayerNorm(fully_connected_dim, eps=layernorm_eps, device=device)
        self.layernorm3 = nn.LayerNorm(fully_connected_dim, eps=layernorm_eps, device=device)

        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout3 = nn.Dropout(dropout_rate)
        self.dropout_ffn = nn.Dropout(dropout_rate)

        self.cross_attend = cross_attend
        if cross_attend:
            self.mha2 = nn.MultiheadAttention(embedding_dim, num_heads, dropout=dropout_rate,
                                              batch_first=True, device=device)
            self.layernorm2 = nn.LayerNorm(fully_connected_dim, eps=layernorm_eps, device=device)
            self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x, enc_output, dec_attention_mask, dec_padding_mask, enc_padding_mask):
        attn1, _ = self.mha1(x, x, x, key_padding_mask=dec_padding_mask, attn_mask=dec_attention_mask)
        attn1 = self.dropout1(attn1)
        out1 = self.layernorm1(attn1 + x)
        
        if self.cross_attend:
            attn2, _ = self.mha2(out1, enc_output, enc_output, key_padding_mask=enc_padding_mask)
            attn2 = self.dropout2(attn2)
            out2 = self.layernorm2(attn2 + out1)
        else:
            out2 = out1

        ffn_output = self.ffn(out2)
        ffn_output = self.dropout_ffn(ffn_output)
        out3 = self.layernorm3(ffn_output + out2)
        return out3


class Decoder(nn.Module):
    def __init__(self, num_layers: list, embedding_dim: int, num_heads: int,
                 fully_connected_dim: int, num_items: list,
                 maximum_position_encoding: int, cross_attend=True,
                 dropout_rate: float=0.1, layernorm_eps: float=1e-6, device: str='cpu'):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        
        self.embeddings = nn.ParameterList(
            [
                nn.Embedding(n_item, self.embedding_dim, padding_idx=n_item-1, device=device)
                for n_item in num_items
            ]
        )
        self.pos_encoding = positional_encoding(maximum_position_encoding, self.embedding_dim, device)

        self.dec_layers = nn.ParameterList(
            [DecoderLayer(embedding_dim=self.embedding_dim,
                          num_heads=num_heads,
                          fully_connected_dim=fully_connected_dim,
                          device=device,
                          cross_attend=cross_attend,
                          dropout_rate=dropout_rate,
                          layernorm_eps=layernorm_eps)
                for _ in range(self.num_layers)])
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, enc_output, dec_attention_mask, dec_padding_mask, enc_padding_mask):
        seq_len = x.shape[1]

        x = torch.concat(
            [self.embeddings[0](x[:, :1])] + [self.embeddings[i](x[:, i+1:i+2]) for i in range(seq_len-1)],
            axis=1
        )
        x *= torch.sqrt(torch.tensor(self.embedding_dim))
        x[:,0] = enc_output[:, 0]
        x += self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x)
        
        for i in range(self.num_layers):
            x = self.dec_layers[i](
                x, enc_output, dec_attention_mask, dec_padding_mask, enc_padding_mask)

        return x


class Generator(nn.Module):
    def __init__(self, num_items: int, num_items_out: list,
                 num_layers: int = 6, num_heads: int = 16,
                 embedding_dim: int = 256, hidden_size: int = 256,
                 dropout_rate: float = 0.1,
                 max_item_list_length: int = 5,
                 device: str = 'cpu') -> None:
        super().__init__()

        self.embedding_dim = embedding_dim
        self.device = device

        self.decoder = Decoder(
            num_layers, embedding_dim=embedding_dim,
            num_items=num_items, num_heads=num_heads,
            fully_connected_dim=hidden_size, cross_attend=False,
            maximum_position_encoding=max_item_list_length,
            dropout_rate=dropout_rate, device=device,
        )

        self.classifier = nn.ParameterList(
            [
                nn.Linear(embedding_dim, num, device=device)
                for num in num_items_out
            ]
        )

        # attention mask (since pos length is fixed)
        self.attention_mask = create_look_ahead_mask(max_item_list_length)

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

    def fit(self, data_loader, criterion, optimizer, batch_size=512, epoches=100, verbose=1):
        if verbose > 0:
            if verbose > 2:
                verbose_epoches = 1
            elif verbose > 1:
                verbose_epoches = 10
            else:
                verbose_epoches = 20

        for epoch in range(epoches):
            self.train()
            loss_hist = []
            for i, batch in enumerate(tqdm(data_loader, disable=(verbose<4))):
                inputs = torch.cat(
                    [
                        torch.tensor([0]*batch.shape[0]).view(-1, 1),
                        batch[:, :-1],
                    ],
                    axis=1
                ).to(self.device)

                noises = torch.randn(inputs.shape[0], 1, self.embedding_dim).to(self.device)
                logits = self.forward(inputs, noises, None, None, self.attention_mask)

                loss = 0
                for i, l in enumerate(logits):
                    loss += criterion(l, batch[:, i])
                loss_hist.append(loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if verbose > 0:
                if epoch % verbose_epoches == 0:
                    print(f'epoch {epoch+1}/{epoches}: loss: {sum(loss_hist)/len(loss_hist)}')
        if verbose > 0:
            print(f'loss: {sum(loss_hist)/len(loss_hist)}')
    
    def generate(self, noises):
        self.eval()

        outputs = torch.tensor([], dtype=torch.long).to(self.device)
        for i in range(5):
            inputs = torch.cat([torch.tensor([0]*noises.shape[0]).view(-1, 1).to(self.device), outputs], axis=1)
            with torch.no_grad():
                logits = self.forward(inputs, noises, None, None, None)
            predictions = torch.argmax(logits[-1], dim=-1)
            outputs = torch.cat([outputs, predictions[:].view(-1, 1)], dim=-1)
        return outputs
