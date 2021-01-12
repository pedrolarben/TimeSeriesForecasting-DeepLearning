import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.scale = nn.Parameter(torch.ones(1))

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.scale * self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerDecoderConvModel(nn.Module):
    def __init__(self, src_dim, tgt_dim, d_model=256,nhead=8, num_layers=3,k = 1, dropout=0.1):
        super(TransformerDecoderConvModel, self).__init__()

        self.d_model = d_model
        
        self.encoder = nn.Linear(src_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        self.decoder = nn.Linear(tgt_dim, d_model)
        self.pos_decoder = PositionalEncoding(d_model, dropout)

        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead,dim_feedforward=d_model*4, dropout=dropout, activation='relu')
        decoder_layer.self_attn = CausalMultiHeadedAttention(d_model=d_model, h=nhead,k=k, dropout=dropout)
        decoder_layer.multihead_attn = CausalMultiHeadedAttention(d_model=d_model, h=nhead,k=k, dropout=dropout)
        self.transformer_decoder  = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, tgt_dim)

        self.src_mask = None
        self.trg_mask = None
        self.memory_mask = None

    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz), 1)
        mask = mask.masked_fill(mask==1, float('-inf'))
        return mask

    def forward(self, src, trg):
        src = src.permute(1,0,2)
        trg = trg.permute(1,0,2)

        if self.trg_mask is None or self.trg_mask.size(0) != len(trg):
            self.trg_mask = self.generate_square_subsequent_mask(len(trg)).to(trg.device)



        src = self.encoder(src)
        src = self.pos_encoder(src)

        trg = self.decoder(trg)
        trg = self.pos_decoder(trg)

        output = self.transformer_decoder (trg,src, tgt_mask=self.trg_mask, memory_mask=self.src_mask)
        output = self.fc_out(output)

        return output.permute(1,0,2)

class CausalConv1d(torch.nn.Conv1d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 groups=1,
                 bias=True):
        self.__padding = (kernel_size - 1) * dilation

        super(CausalConv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=self.__padding,
            dilation=dilation,
            groups=groups,
            bias=bias)

    def forward(self, input):
        result = super(CausalConv1d, self).forward(input)
        if self.__padding != 0:
            return result[:, :, :-self.__padding]
        return result


class CausalMultiHeadedAttention(nn.Module):
    def __init__(self, d_model, h ,k, dropout=0.1):
        "Take in model size and number of heads."
        super(CausalMultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.p = dropout
        self.linear1 = nn.Linear(d_model, d_model)
        self.linear2 = nn.Linear(d_model, d_model)
        self.causalConv1d1 = CausalConv1d(d_model, d_model,k)
        self.causalConv1d2 = CausalConv1d(d_model, d_model,k)
        self.attn = None

        
    def forward(self, query, key, value, attn_mask=None,key_padding_mask=None):
        "Implements Figure 2"
        query = query.transpose(0,1)
        key = key.transpose(0,1)
        value = value.transpose(0,1)

        if attn_mask is not None:
            # Same mask applied to all h heads.
            attn_mask = attn_mask.unsqueeze(0)
            attn_mask = attn_mask.unsqueeze(0)

        nbatches = query.size(0)
        
        query = self.causalConv1d1(query.transpose(2,1)).transpose(2,1).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        key = self.causalConv1d2(key.transpose(2,1)).transpose(2,1).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)

        value = self.linear1(value).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)

        # 2) Apply attention on all the projected vectors in batch. 

        x, self.attn = attention(query, key, value, mask=attn_mask, dropout=self.p) 
        
        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linear2(x).transpose(0,1)


def attention(query, key, value, mask=None, dropout=0.0):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)

    p_attn = F.dropout(p_attn, p=dropout)
    return torch.matmul(p_attn, value), p_attn