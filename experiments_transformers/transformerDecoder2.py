import torch
import torch.nn as nn
import pytorch_lightning as pl

from utils import PositionalEncoding, generate_square_subsequent_mask

class TransformerDecoderModel2(pl.LightningModule):
    '''
    Autoregresive Decoder-Only Transformer. Variant number 2.
    '''
    def __init__(self,input_size,output_size, n_features, d_model=256,nhead=8, num_layers=3, dropout=0.1):
        super(TransformerDecoderModel2, self).__init__()

        self.d_model = d_model
        self.criterion = nn.L1Loss()
        self.warmup_steps = 4000

        self.output_size = output_size
        self.n_features = n_features



        self.encoder = nn.Linear(n_features, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        decoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,dim_feedforward=d_model*4, dropout=dropout, activation='relu')
        self.transformer_decoder  = nn.TransformerEncoder(decoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, n_features)

        self.src_mask = None
        
    def forward(self, src):

        src = src.permute(1,0,2)

        if self.src_mask is None or self.src_mask.size(0) != len(src):
            self.src_mask = generate_square_subsequent_mask(len(src)).to(src.device)

        src = self.encoder(src)
        src = self.pos_encoder(src)

        output = self.transformer_decoder(src,self.src_mask)
        output = self.fc_out(output)

        return output.permute(1,0,2)

    def training_step(self, batch, batch_idx):
        x,y = batch
        z = torch.cat((x,y[:,:-1]),1)
        
        y_hat = self(z)
        y_hat = y_hat[:,x.shape[1]:]
        loss = self.criterion(y_hat, y[:, 1:])
        self.log("train_loss",loss,on_epoch=True,logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        z = x
        for i in range(0,self.output_size):
            out = self(z)
            z = torch.cat((z,out[:,-1].unsqueeze(-1).detach()),1) 
            
        y_hat = z[:,x.shape[1]:]
        loss = self.criterion(y_hat, y)
        self.log('val_loss', loss,on_epoch=True,prog_bar=True,logger=True)
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
