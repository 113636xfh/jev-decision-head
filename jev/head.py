"""Set encoder. Blocks and input backbone layers are independent variables."""
import torch
from torch import nn

class Head(nn.Module):
    def __init__(self, n_layers, hc=4, n_embd=2560, d=512, blocks=2, dropout=0.1, fusion='concat'):
        super().__init__()
        if d % hc or d % 8 or blocks < 1 or fusion not in ('concat', 'weighted'):
            raise ValueError('invalid head geometry/fusion')
        self.hc, self.n_embd, self.d, self.fusion = hc, n_embd, d, fusion
        self.proj = nn.Linear(n_embd, d // hc)
        if fusion == 'concat':
            self.fuse = nn.Linear(n_layers * d, d)
        else:
            self.layer_weights = nn.Parameter(torch.zeros(n_layers))
            self.fuse = nn.Linear(d, d)
        self.pool = nn.Linear(2 * d, d)
        self.role = nn.Embedding(3, d)
        layer = nn.TransformerEncoderLayer(d, 8, 4096, dropout, activation='gelu', batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, blocks, norm=nn.LayerNorm(d))
        self.Wq, self.Wo = nn.Linear(d, 512, bias=False), nn.Linear(d, 512, bias=False)
        self.u = nn.Linear(d, 1, bias=False)

    def forward(self, X, counts):
        b, spans, pools, layers, dim = X.shape
        if pools != 2 or dim != self.hc * self.n_embd:
            raise ValueError('bad feature geometry')
        h = self.proj(X.reshape(b, spans, pools, layers, self.hc, self.n_embd))
        h = h.reshape(b, spans, pools, layers, self.d)
        if self.fusion == 'concat':
            h = self.fuse(h.permute(0, 1, 2, 4, 3).reshape(b, spans, pools, layers * self.d))
        else:
            h = self.fuse((h * self.layer_weights.softmax(0)[None, None, None, :, None]).sum(3))
        h = self.pool(h.reshape(b, spans, pools * self.d))
        batch = torch.arange(b, device=X.device)
        max_k = int(counts.max().item())
        seq = torch.cat([h[batch, counts + 1][:, None], h[:, :1], h[:, 1:max_k + 1]], dim=1)
        roles = torch.tensor([2, 0] + [1] * max_k, device=X.device)
        mask = torch.arange(max_k + 2, device=X.device)[None] < counts[:, None] + 2
        seq = self.enc(seq + self.role(roles)[None], src_key_padding_mask=~mask)
        logits = (self.Wq(seq[:, :1]) * self.Wo(seq[:, 2:])).sum(-1) / (512 ** 0.5) + self.u(seq[:, 2:]).squeeze(-1)
        return logits.masked_fill(~mask[:, 2:], float('-inf'))
