import tensorflow as tf
import pickle
import numpy as np
import os 
import argparse

from dataGenerator import dataGenerator
from ratSimulator import RatSimulator
from agent import Network
from trainer import Trainer
from showGridCells import showGridCells
#HYPERPARAMETERS
LSTMUnits=128
linearLayerUnits=512
PlaceCells_units=256
HeadCells_units=12

learning_rate=1e-5
clipping=1e-5
weightDecay=1e-5
batch_size=10
SGDSteps=300000
numberSteps=800
num_features=3 #num_features=[velocity, sin(angVel), cos(angVel)]

bins=32

#Number of trajectories to generate and use as data to train the network
num_trajectories=500

#Number of trajectories used to display the activity maps
showCellsTrajectories=100

#Initialize place cells centers and head cells centers. Every time the program starts they are the same
rs=np.random.RandomState(seed=10)
#Generate 256 Place Cell centers
place_cell_centers=rs.uniform(0, 2.2 ,size=(PlaceCells_units,2))
#Generate 12 Head Direction Cell centers
head_cell_centers=rs.uniform(-np.pi,np.pi,size=(HeadCells_units))

#Class that generate the trajectory and allows to compute the Place Cell and Head Cell distributions
dataGenerator=dataGenerator(numberSteps, num_features, PlaceCells_units, HeadCells_units)

global_step=0

def prepareTestingData():
    if os.path.isfile("trajectoriesDataTesting.pickle"):
        print("\nLoading test data..")
        fileData=pickle.load(open("trajectoriesDataTesting.pickle","rb"))
        inputDataTest=fileData['X']
        posTest=fileData['pos']
        angleTest=fileData['angle']
    #Create new data
    else:
        print("\nCreating test data..")
        inputDataTest, posTest, angleTest=dataGenerator.generateData(batch_size=10)
        mydict={"X":inputDataTest,"pos":posTest, "angle":angleTest}
        with open('trajectoriesDataTesting.pickle', 'wb') as f:
            pickle.dump(mydict, f)

    init_LSTMStateTest=np.zeros((10,8,PlaceCells_units + HeadCells_units))
    for i in range(8):
        init_LSTMStateTest[:, i, :PlaceCells_units]=dataGenerator.computePlaceCellsDistrib(posTest[:,(i*100)], place_cell_centers)
        init_LSTMStateTest[:, i, PlaceCells_units:]=dataGenerator.computeHeadCellsDistrib(angleTest[:,(i*100)], head_cell_centers)

    return inputDataTest, init_LSTMStateTest, posTest

def trainAgent(agent):
    global global_step

    trainer=Trainer(agent, PlaceCells_units, numberSteps)

    #Load testing data
    inputDataTest,init_LSTMStateTest, posTest=prepareTestingData()

    while (global_step<SGDSteps):
        #Create training Data
        inputData, pos, angle=dataGenerator.generateData(batch_size=num_trajectories)

        labelData=np.zeros((num_trajectories, numberSteps, PlaceCells_units + HeadCells_units))
        for t in range(numberSteps):
            labelData[:,t, :PlaceCells_units]=dataGenerator.computePlaceCellsDistrib(pos[:,t], place_cell_centers)
            labelData[:,t, PlaceCells_units:]=dataGenerator.computeHeadCellsDistrib(angle[:,t], head_cell_centers)

        for startB in range(0, num_trajectories, batch_size):
            endB=startB+batch_size
            #return a tensor of shape 10,800,3
            batchX=inputData[startB:endB]
            #return a tensor of shape 10,800,256+12
            batchY=labelData[startB:endB]

            trainer.training(batchX,batchY, global_step)
            
            if (global_step%800==0):
                print("\n>>Testing the agent")
                trainer.testing(inputDataTest, init_LSTMStateTest, posTest, place_cell_centers, global_step)

                print(">>Global step:",global_step,"Saving the model..\n")
                agent.save_restore_Model(restore=False, epoch=global_step)

            global_step+=8

if __name__ == '__main__':
    parser=argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("mode", help="train - will train the model \nshowcells - will create the activity maps of the neurons")
    args=parser.parse_args()
    with tf.Session() as sess:
        try:
            agent=Network(sess, lr=learning_rate, hu=LSTMUnits, lu=linearLayerUnits, place_units=PlaceCells_units, head_units=HeadCells_units, clipping=clipping, weightDecay=weightDecay, batch_size=batch_size, num_features=num_features, n_steps=numberSteps)
            if os.path.exists("agentBackup"):
                print("Loading the model..")
                agent.save_restore_Model(restore=True)
                print("Model updated at global step:", sess.run(agent.epoch), "loaded")
                global_step=sess.run(agent.epoch)


            if(args.mode=="train"):    
                trainAgent(agent)
            elif(args.mode=="showcells"):
                showGridCells(agent, dataGenerator, showCellsTrajectories, numberSteps, PlaceCells_units, HeadCells_units, linearLayerUnits,
                bins,place_cell_centers, head_cell_centers)

                
        except (KeyboardInterrupt,SystemExit):
            print("\n\nProgram shut down, saving the model..")
            agent.save_restore_Model(restore=False, epoch=global_step)
            print("\n\nModel saved!\n\n")
            raise