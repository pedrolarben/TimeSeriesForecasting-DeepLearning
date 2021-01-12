import torch
import torch.nn as nn
import math



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

class TransformerDecoderModel(nn.Module):
    def __init__(self, src_dim, tgt_dim, d_model=256,nhead=8, num_layers=3, dropout=0.1):
        super(TransformerDecoderModel, self).__init__()

        self.d_model = d_model
        
        self.encoder = nn.Linear(src_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        self.decoder = nn.Linear(tgt_dim, d_model)
        self.pos_decoder = PositionalEncoding(d_model, dropout)

        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead,dim_feedforward=d_model*4, dropout=dropout, activation='relu')
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

        output = self.transformer_decoder(trg,src, tgt_mask=self.trg_mask, memory_mask=self.src_mask)
        output = self.fc_out(output)

        return output.permute(1,0,2)