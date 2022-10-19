from torch import squeeze
import wandb
def BinaryClassiferTrainingLoop(Device,DataLoader,Model,LossFn,Optimizer):
    Batches = len(DataLoader)
    Model.train()
    Running_Loss = 0
    for xx, yy in DataLoader:

        BatchSize = xx.shape[0]
        ImageSize = xx.shape[1]

        #Images = Data['image'].float().to(Device)
        #Modes = Data['mode'].to(Device)

        xx = xx.reshape((BatchSize,1,ImageSize,ImageSize)).float().to(Device)
        yy = yy.to(Device)

        Optimizer.zero_grad()

        #Outputs = squeeze(Model(Images))
        out = Model(xx)

        #Loss = LossFn(Outputs, Modes)
        Loss = LossFn(out,yy)

        Loss.backward()
        Optimizer.step()

        Running_Loss += Loss.item()
        #wandb.log({'Training Batch': Batch})
            
    return Running_Loss/Batches  