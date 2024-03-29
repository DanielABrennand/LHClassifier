###
#Imports
###

from torch import nn,optim,save,cuda,from_numpy,manual_seed
from torch.utils.data import DataLoader,TensorDataset
from torchvision import transforms
from os.path import join
from numpy import load,random
import gc
import time
from simvue import Run

#from UtilityFunctions import LogitPercentConverter,H5ToNumpy as LPC,H2N

RunTime = time.time()

###
#Globals/Constants
###

#Overall Info
PROJECT = "LHClassifierGeoffTimeTest"
MODEL_NAME = "AlexNetV2"

DEVICE = "cuda" if cuda.is_available() else "cpu"

#Hyper Parameters
SHUFFLE = True
WORKERS = 0
BATCH_SIZE = 20

EPOCHS = 1
LOSS_FN = "BCELoss"
OPTIMIZER = "SGD"
MOMENTUM = 0.9
LEARNING_RATE = 0.001

CONFIDENCE_THRESHOLD = 0.8

SEED = None
if not SEED:
    SEED = random.randint(0,9223372036854775807)

#Data Locations
TRAINING_DATA_ROOT = "/home/dbren/VSCode/DataStore/RBB_FILES/Training"
TRAINING_MODES_PATH = "/home/dbren/VSCode/DataStore/RBB_FILES/Training/TrainingModes.csv"

VALIDATION_DATA_ROOT = "/home/dbren/VSCode/DataStore/RBB_FILES/Validation"
VALIDATION_MODES_PATH = "/home/dbren/VSCode/DataStore/RBB_FILES/Validation/ValidationModes.csv"

TESTING_DATA_ROOT = "/home/dbren/VSCode/DataStore/RBB_FILES/Testing"
TESTING_MODES_PATH = "/home/dbren/VSCode/DataStore/RBB_FILES/Testing/TestingModes.csv"

IMAGE_SIZE = 512
#Optional Modes
EPOCH_SAVE_INTERVAL = 0 #0 for off
FINAL_SAVING = True

TESTING = True

#Outputs
HEURISTICS_SAVE_PATH = "/home/dbren/VSCode/DataStore/Heuristics"

SIMVUE_FOLDER = "/Classifier"

###
#WandB setup
###

configuration = {"Project": PROJECT,
                 "Model": MODEL_NAME,
                 "Epochs": EPOCHS,
                 "Batch Size": BATCH_SIZE,
                 "Optimizer": OPTIMIZER,
                 "Loss Function": LOSS_FN,
                 "Learning Rate": LEARNING_RATE,
                 "Momentum": MOMENTUM,
                 "Device" : DEVICE,
                 "Epoch Save Interval": "Off" if EPOCH_SAVE_INTERVAL == 0 else EPOCH_SAVE_INTERVAL,
                 "Image Size": IMAGE_SIZE,
                 "Seed" : SEED,
                 "Confidence Threshold": CONFIDENCE_THRESHOLD
                }

run = Run()

run.init(metadata=configuration,folder=SIMVUE_FOLDER)

manual_seed(SEED)

###
#Model
###

from Models import AlexNet
Net = AlexNet().to(DEVICE)

###
#Data input
###

from DataSets import H5DataSet

TrainingDataSet = H5DataSet(TRAINING_DATA_ROOT,TRAINING_MODES_PATH)
TrainLoader = DataLoader(TrainingDataSet, batch_size=BATCH_SIZE, shuffle=SHUFFLE, num_workers=WORKERS)

ValidationDataSet = H5DataSet(VALIDATION_DATA_ROOT,VALIDATION_MODES_PATH)
ValidationLoader = DataLoader(ValidationDataSet, batch_size=BATCH_SIZE, shuffle=SHUFFLE, num_workers=WORKERS)

TestingDataSet = H5DataSet(TESTING_DATA_ROOT,TESTING_MODES_PATH)
TestLoader = DataLoader(TestingDataSet, batch_size=BATCH_SIZE, shuffle=False, num_workers=WORKERS)

###
#Training loop
###

from TrainingLoops import BinaryClassiferTrainingLoop
from TestingLoops import BinaryClassiferTestingLoop

#LossFn = nn.BCELoss()
LossFn = nn.BCEWithLogitsLoss()
Optimizer = optim.SGD(Net.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
#Optimizer = optim.Adam(Net.parameters(),lr=LEARNING_RATE)

for Epoch in range(EPOCHS):
    TrainingLoss = BinaryClassiferTrainingLoop(DEVICE,TrainLoader,Net,LossFn,Optimizer)
    ValidationLoss,Correct = BinaryClassiferTestingLoop(DEVICE,ValidationLoader,Net,LossFn,CONFIDENCE_THRESHOLD)

    run.log_metrics({'Train Loss': TrainingLoss, 
               'Validation Loss': ValidationLoss,
               'Validation Correct': Correct})

    if EPOCH_SAVE_INTERVAL:
        if Epoch%EPOCH_SAVE_INTERVAL == 0:
            save(Net.state_dict(), join(HEURISTICS_SAVE_PATH,(PROJECT + "_" + run.name + "_Epoch_" + str(Epoch) + ".pth")))

TVTime = time.time() - RunTime

del TrainingDataSet, TrainLoader, ValidationDataSet, ValidationLoader
gc.collect()

###
#Final Testing Loop
###

if TESTING:
    TestingLoss,Correct = BinaryClassiferTestingLoop(DEVICE,TestLoader,Net,LossFn,CONFIDENCE_THRESHOLD,True)
    run.log_metrics({'Final Test Loss':TestingLoss,
                'Final Test Correct': Correct})

TVTTime = time.time() - RunTime

###
#Outputs
###

run.log_metrics({'TV time':TVTime,
            'TVTTime':TVTTime})


if FINAL_SAVING:
    save(Net.state_dict(), join(HEURISTICS_SAVE_PATH,("{}_{}_Full.pth").format(PROJECT,run.name)))
