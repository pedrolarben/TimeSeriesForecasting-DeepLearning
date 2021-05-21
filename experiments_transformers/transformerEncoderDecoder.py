import torch
import torch.nn as nn
import pytorch_lightning as pl
from utils import PositionalEncoding, generate_square_subsequent_mask

class transformerEncoderDecoder(pl.LightningModule):
    '''
    Full Transformer
    '''
    def __init__(self,input_size,output_size, n_features, d_model=256,nhead=8, num_layers=3, dropout=0.1):
        super(transformerEncoderDecoder, self).__init__()

        self.d_model = d_model
        self.criterion = nn.L1Loss()
        self.warmup_steps = 4000
        
        self.encoder = nn.Linear(src_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        self.decoder = nn.Linear(tgt_dim, d_model)
        self.pos_decoder = PositionalEncoding(d_model, dropout)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,dim_feedforward=d_model*4 , dropout=dropout, activation='relu')
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)  

        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead,dim_feedforward=d_model*4, dropout=dropout, activation='relu')
        self.transformer_decoder  = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, tgt_dim)

        self.src_mask = None
        self.trg_mask = None
        self.memory_mask = None
        
    def forward(self, src,trg):

        src = src.permute(1,0,2)
        trg = trg.permute(1,0,2)

        if self.trg_mask is None or self.trg_mask.size(0) != len(trg):
            self.trg_mask = generate_square_subsequent_mask(len(trg)).to(trg.device)



        src = self.encoder(src)
        src = self.pos_encoder(src)

        trg = self.decoder(trg)
        trg = self.pos_decoder(trg)

        memory = self.transformer_encoder(src,self.src_mask)

        output = self.transformer_decoder(trg,memory, tgt_mask=self.trg_mask, memory_mask=self.src_mask)
        output = self.fc_out(output)

        return output.permute(1,0,2)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y = torch.cat((x[:,-1,:].unsqueeze(1),y ),1)
        
        y_hat = self(x,y[:, :-1])
        loss = self.criterion(y_hat, y[:, 1:])
        self.log("train_loss",loss,on_epoch=True,logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        decoderInput = x[:, -1].unsqueeze(-1)
        for i in range(0,self.output_size):
            out = self(x,decoderInput)
            decoderInput = torch.cat((decoderInput,out[:,-1].unsqueeze(-1).detach()),1) 
            
        y_hat = decoderInput[:, 1:]
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
