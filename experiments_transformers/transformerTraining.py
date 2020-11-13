import torch
from torch.autograd import Variable
import numpy as np
def loss_backprop(generator, criterion, out, targets):
    """
    Memory optmization. Compute each timestep separately and sum grads.
    """
    assert out.size(1) == targets.size(1)
    total = 0.0
    out_grad = []

    for i in range(out.size(1)):

        out_column = Variable(out[:, i].data, requires_grad=True)
        gen = generator(out_column)
        loss = criterion(gen, targets[:, i])
        total += float(loss.data.item())
        loss.backward()
        out_grad.append(out_column.grad.data.clone())

    out_grad = torch.stack(out_grad, dim=1)
    out.backward(gradient=out_grad)
    return total/out.size(1)

def loss_validation(generator, criterion, out, targets):

    assert out.size(1) == targets.size(1)
    total = 0.0
  

    for i in range(out.size(1)):

        out_column = Variable(out[:, i].data, requires_grad=True)
        gen = generator(out_column)
        loss = criterion(gen, targets[:, i])
        total += float(loss.data.item())

    return total/out.size(1)

def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

def make_std_mask(tgt):
    tgt_mask = Variable(subsequent_mask(tgt.size(-2)).type_as(tgt.data))
    return tgt_mask

def train_epoch(train_iter, model, criterion, model_opt):
    model.train()
    totalLoss = 0
    for i, batch in enumerate(train_iter):
        src, trg, trg_mask = \
            batch.src, batch.trg, batch.trg_mask
        out = model.forward(src, trg[:, :-1], None, trg_mask[:, :-1, :-1])

        loss = loss_backprop(model.module.generator, criterion, out, trg[:, 1:])  #model.module need to use generator throught wrapper nn.DataParallel
        totalLoss += float(loss)
        model_opt.step()
        model_opt.optimizer.zero_grad()
        #if i % 5 == 1:
        #print(i,"Batch loss", loss)
    print("Train mean loss",totalLoss/(i+1))
    return totalLoss/(i+1)

def singleStepvalidEpoch(valid_iter, model, criterion):
    model.eval()
    totalLoss = 0
    with torch.no_grad():
      for i, batch in enumerate(valid_iter):
          src, trg, trg_mask = \
              batch.src, batch.trg, batch.trg_mask
          out = model.forward(src, trg[:, :-1], None, trg_mask[:, :-1, :-1])
          loss = loss_validation(model.module.generator, criterion, out, trg[:, 1:]) 
          totalLoss += loss
          #print(i,"Batch validation loss", loss)
    print("Single Step Validation loss:" , totalLoss/(i+1))


def multiStepValidEpoch(valid_iter, model, criterion):
    model.eval()
    totalLoss = 0
    with torch.no_grad():
      for i, batch in enumerate(valid_iter):
          src, trg = \
              batch.src, batch.trg
          nSteps = trg.shape[1] - 1
          decoderInput = trg[:, 0]
          decoderInput = decoderInput.unsqueeze(-1)
          for _ in range(nSteps): # We append the prediction on step 0 to the original input to get the input for step 1 and so on.
            out = model.forward(src,decoderInput , None, None)  # We don't need mask for evaluation, we are not giving the model any future input.
 
            gen = model.module.generator(out[:,-1]).detach() #detach() is quite important, otherwise we will keep the variable "gen" in memory and cause an out of memory error.
            gen = gen.unsqueeze(-1)
                
            decoderInput = torch.cat((decoderInput,gen),1) 

          output = decoderInput[:, 1:]

          loss = criterion(output,trg[:, 1:])

          totalLoss += float(loss) #float() is quite important, otherwise we will keep "loss" and its gradient in memory and cause an out of memory error.
          #print(i,"Batch validation loss", loss)
    print("Multi Step Validation loss:" , totalLoss/(i+1))
    return totalLoss/(i+1)

class Batch:
        def __init__(self, src, trg, trg_mask, ntokens):
            self.src = src
            self.trg = trg
            self.trg_mask = trg_mask
            self.ntokens = ntokens
    
def getBatches(x,y,batchSize):

    y = torch.cat((x[:,-1,:],y ),1) # We put the last element od the encoders input as the first element of the decoder input.
                                    # This imitates the function of the start token in nlp

    y = y.unsqueeze(-1)
    permutation = torch.randperm(x.shape[0])
    for i in range(0, len(x), batchSize):
        seqLen = min(batchSize, len(x) - i)
        indices = permutation[i:i+seqLen]
        src = Variable(x[indices], requires_grad = False)
        tgt = Variable(y[indices], requires_grad = False)
        tgt_mask = make_std_mask(tgt)
        yield Batch(src, tgt, tgt_mask, seqLen)




def train(model,x_train,y_train,x_test,y_test,epochs,criterion,model_opt,batchSize):
    
    for e in range(epochs):
        trainLoss = train_epoch(getBatches(x_train,y_train,batchSize), model, criterion, model_opt)
        valLoss = multiStepValidEpoch(getBatches(x_test,y_test,batchSize), model, criterion)
        #singleStepvalidEpoch(getBatches(x_test,y_test,batchSize), model, criterion)
    return trainLoss,valLoss


def getBatchesEval(x,batchSize):

  permutation = torch.randperm(x.shape[0])

  for i in range(0, len(x), batchSize):
    seqLen = min(batchSize, len(x) - i)

    indices = permutation[i:i+seqLen]
    src = Variable(x[indices], requires_grad = False)
    
    yield Batch(src,None,None, seqLen)

def predictMultiStep(x_test, model, nSteps,batchSize):
    model.eval()
    with torch.no_grad():
      test_iter = getBatchesEval(x_test,batchSize)
      pred = []
      for i, batch in enumerate(test_iter):
          src = batch.src
          nSteps = nSteps
          decoderInput = src[:, -1]
          decoderInput = decoderInput.unsqueeze(-1)
          for _ in range(nSteps): # We append the prediction on step 0 to the original input to get the input for step 1 and so on.
            out = model.forward(src,decoderInput , None, None)  # We don't need mask for evaluation, we are not giving the model any future input.
            
            gen = model.module.generator(out[:,-1]).detach() #detach() is quite important, otherwise we will keep the variable "gen" in memory and cause an out of memory error.
            gen = gen.unsqueeze(-1)
                
            decoderInput = torch.cat((decoderInput,gen),1) 

          batchPred = decoderInput[:, 1:]
          pred.extend(batchPred.reshape(batchPred.shape[0]*batchPred.shape[1]).to("cpu").detach().numpy())
    return pred





