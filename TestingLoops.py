from torch import no_grad, squeeze, float
import wandb
def BinaryClassiferTestingLoop(Device,DataLoader,Model,LossFn):
    Size = len(DataLoader.dataset)
    Batches = len(DataLoader)
    Model.eval()
    TestLoss, Correct = 0, 0
    with no_grad():
        for xx, yy in DataLoader:

            BatchSize = xx.shape[0]
            ImageSize = xx.shape[1]

            #Images = Data['image'].float().to(Device)
            #Modes = Data['mode'].to(Device)

            xx = xx.reshape((BatchSize,1,ImageSize,ImageSize)).float().to(Device)
            yy =yy.to(Device)

            #Outputs = squeeze(Model(Images))
            out = Model(xx)

            #TestLoss += LossFn(Outputs,Modes).item()
            TestLoss  += LossFn(out,yy).item()

            #Correct += (Outputs.argmax(1) == Modes).type(float).sum().item()
            Correct  += (out.argmax(1) == yy).type(float).sum().item()

            #wandb.log({'Testing Batch': Batch})
        TestLoss /= Batches
        Correct /= Size

    return TestLoss,Correct