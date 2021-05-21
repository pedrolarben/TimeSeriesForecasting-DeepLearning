import torch
import torch.nn as nn
import pytorch_lightning as pl
from utils import PositionalEncoding


class TransformerEncoderModel(pl.LightningModule):
    '''
    Non-Autoregresive encoder Transformer + MLP head
    '''
    def __init__(self,input_size,output_size, n_features, d_model=256,nhead=8, num_layers=3, dropout=0.1):
        super(TransformerEncoderModel, self).__init__()

        self.d_model = d_model
        self.criterion = nn.L1Loss()
        self.warmup_steps = 4000

        self.output_size = output_size
        self.n_features = n_features



        self.encoder = nn.Linear(n_features, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        decoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,dim_feedforward=d_model*4, dropout=dropout, activation='relu')
        self.transformer_decoder  = nn.TransformerEncoder(decoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model*input_size, output_size*n_features)

        self.src_mask = None
        
    def forward(self, src):

        src = self.encoder(src)
        src = self.pos_encoder(src)

        src = src.permute(1,0,2)
        output = self.transformer_decoder(src)
        output = output.permute(1,0,2)

        output = torch.flatten(output,1)
        output = self.fc_out(output)

        return output.view(-1,self.output_size,self.n_features)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log("train_loss",loss,on_epoch=True,logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('val_loss', loss,on_epoch=True,prog_bar=True,logger=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), betas=(0.9, 0.98), eps=1e-9)
        return optimizer
    
    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure,
        on_tpu=False, using_native_amp=False, using_lbfgs=False):

        d_model, steps, warmup_steps = self.d_model, self.global_step + 1, self.warmup_steps
        rate = (d_model ** (- 0.5)) * min(steps ** (- 0.5), steps * warmup_steps ** (- 1.5))

        for pg in optimizer.param_groups:
            pg['lr'] = rate

        optimizer.step(closure=optimizer_closure)
