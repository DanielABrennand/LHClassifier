###
#Imports
###

from torch import nn,optim,save,cuda,from_numpy,manual_seed
from torch.utils.data import DataLoader,TensorDataset
from torchvision import transforms
from os.path import join
from numpy import load,random
import wandb
import gc

#from UtilityFunctions import LogitPercentConverter,H5ToNumpy as LPC,H2N

###
#Globals/Constants
###

#Overall Info
PROJECT = "LHClassifierGeoffSGDNetAdamTest1"
MODEL_NAME = "AlexNetV1"

DEVICE = "cuda" if cuda.is_available() else "cpu"

#Hyper Parameters
SHUFFLE = True
WORKERS = 0
BATCH_SIZE = 20

EPOCHS = 100
LOSS_FN = "Cross Entropy Loss"
OPTIMIZER = "SGD"
MOMENTUM = 0.9
LEARNING_RATE = 0.001

SEED = random.randint(0,9223372036854775807)

#Data Locations
TRAINING_DATA_ROOT = "/home/dbren/VSCode/DataStore/TrainingFrames.npy"
TRAINING_MODES_PATH = "/home/dbren/VSCode/DataStore/TrainingModes.npy"

VALIDATION_DATA_ROOT = "/home/dbren/VSCode/DataStore/ValidationFrames.npy"
VALIDATION_MODES_PATH = "/home/dbren/VSCode/DataStore/ValidationModes.npy"

TESTING_DATA_ROOT = "/home/dbren/VSCode/DataStore/TestingFrames.npy"
TESTING_MODES_PATH = "/home/dbren/VSCode/DataStore/TestingModes.npy"

IMAGE_SIZE = 512
#Optional Modes
EPOCH_SAVE_INTERVAL = 0 #0 for off
FINAL_SAVING = True

TESTING = True

#Outputs
HEURISTICS_SAVE_PATH = "/home/dbren/VSCode/DataStore/Heuristics"

###
#WandB setup
###

configuration = {"Model": MODEL_NAME,
                 "Epochs": EPOCHS,
                 "Batch Size": BATCH_SIZE,
                 "Optimizer": OPTIMIZER,
                 "Loss Function": LOSS_FN,
                 "Learning Rate": LEARNING_RATE,
                 "Momentum": MOMENTUM,
                 "Device" : DEVICE,
                 "Epoch Save Interval": "Off" if EPOCH_SAVE_INTERVAL == 0 else EPOCH_SAVE_INTERVAL,
                 "Image Size": IMAGE_SIZE,
                 "Seed" : SEED
                }

run = wandb.init(project=PROJECT,
                 notes='',
                 config=configuration)

manual_seed(SEED)

###
#Model
###

from Models import AlexNet
Net = AlexNet().to(DEVICE)

###
#Data input
###

TrainingDataFrames = from_numpy(load(TRAINING_DATA_ROOT))
TrainingDataModes = from_numpy(load(TRAINING_MODES_PATH))
TrainingDataSet = TensorDataset(TrainingDataFrames,TrainingDataModes)
TrainLoader = DataLoader(TrainingDataSet, batch_size=BATCH_SIZE, shuffle=SHUFFLE, num_workers=WORKERS)
del TrainingDataFrames, TrainingDataModes

ValidationDataFrames = from_numpy(load(VALIDATION_DATA_ROOT))
ValidationDataModes = from_numpy(load(VALIDATION_MODES_PATH))
ValidationDataSet = TensorDataset(ValidationDataFrames,ValidationDataModes)
ValidationLoader = DataLoader(ValidationDataSet, batch_size=BATCH_SIZE, shuffle=SHUFFLE, num_workers=WORKERS)
del ValidationDataFrames, ValidationDataModes

TestingDataFrames = from_numpy(load(TESTING_DATA_ROOT))
TestingDataMode = from_numpy(load(TESTING_MODES_PATH))
TestingDataSet = TensorDataset(TestingDataFrames,TestingDataMode)
TestLoader = DataLoader(TestingDataSet, batch_size=BATCH_SIZE, shuffle=SHUFFLE, num_workers=WORKERS)
del TestingDataFrames, TestingDataMode

gc.collect()

###
#Training loop
###

from TrainingLoops import BinaryClassiferTrainingLoop
from TestingLoops import BinaryClassiferTestingLoop

LossFn = nn.CrossEntropyLoss()
Optimizer = optim.SGD(Net.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
#Optimizer = optim.Adam(Net.parameters(),lr=LEARNING_RATE)

for Epoch in range(EPOCHS):
    TrainingLoss = BinaryClassiferTrainingLoop(DEVICE,TrainLoader,Net,LossFn,Optimizer)
    ValidationLoss,Correct = BinaryClassiferTestingLoop(DEVICE,ValidationLoader,Net,LossFn)

    wandb.log({'Train Loss': TrainingLoss, 
               'Validation Loss': ValidationLoss,
               'Validation Correct': Correct})

    if EPOCH_SAVE_INTERVAL:
        if Epoch%EPOCH_SAVE_INTERVAL == 0:
            save(Net.state_dict(), join(HEURISTICS_SAVE_PATH,(PROJECT + "_" + wandb.run.name + "_Epoch_" + str(Epoch) + ".pth")))

del TrainingDataSet, TrainLoader, ValidationDataSet, ValidationLoader
gc.collect()

###
#Testing Loop
###

if TESTING:
    TestingLoss,Correct = BinaryClassiferTestingLoop(DEVICE,TestLoader,Net,LossFn)
    wandb.log({'Final Test Loss':TestingLoss,
                'Final Test Correct': Correct})

###
#Outputs
###

if FINAL_SAVING:
    save(Net.state_dict(), join(HEURISTICS_SAVE_PATH,(PROJECT + "_" + wandb.run.name + "_Full_" + ".pth")))