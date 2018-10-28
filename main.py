import tensorflow as tf
import pickle
import numpy as np
import os 
import argparse

from dataGenerator import dataGenerator
from ratSimulator import RatSimulator
from agent import network
#HYPERPARAMETERS
LSTMUnits=128
linearLayerUnits=512
PlaceCells_units=256
HeadCells_units=12

learning_rate=0.00001
clipping=0.000001
weightDecay=0.00001
batch_size=10
epoches=300000
numberSteps=800
num_features=3 #num_features=[velocity, sin(angVel), cos(angVel)]

#Initialize place cells centers and head cells centers. Every time the program starts they are the same
rs=np.random.RandomState(seed=10)
#Generate 256 Place Cell centers
place_cell_centers=rs.uniform(0, 2.2 ,size=(PlaceCells_units,2))
#Generate 12 Head Direction Cell centers
head_cell_centers=rs.uniform(-np.pi,np.pi,size=(HeadCells_units))

#Number of trajectories to generate and use as data to train the network
trajectories=250
#Class that generate the trajectory and allows to compute the Place Cell and Head Cell distributions
dataGenerator=dataGenerator(trajectories, numberSteps, num_features, PlaceCells_units, HeadCells_units)

global_step=0


command="train"
def trainAgent(agent):
    global global_step
    #Load data
    fileData=pickle.load(open("trajectoriesData.pickle","rb"))
    inputData=fileData['X']
    pos=fileData['pos']
    angle=fileData['angle']

    labelData=np.zeros((pos.shape[0], numberSteps, PlaceCells_units + HeadCells_units))
    for t in range(numberSteps):
        labelData[:,t, :PlaceCells_units]=dataGenerator.computePlaceCellsDistrib(pos[:,t], place_cell_centers)
        labelData[:,t, PlaceCells_units:]=dataGenerator.computeHeadCellsDistrib(angle[:,t], head_cell_centers)
    
    batches=int(inputData.shape[0]/batch_size)

    for epoch in range(epoches):
        print("\n>>Epoch", epoch)
        startB=0
        for batch in range(batches):
            endB=startB+batch_size
            #return a tensor of shape 10,800,3
            batchX=inputData[startB:endB]
            #return a tensor of shape 10,800,256+12
            batchY=labelData[startB:endB]
            #Retrieve data for initialize LSTM states
            #with shape 10,256+12
            batch_initLSTM=batchY[:,0]

            agent.training(batchX,batchY,batch_initLSTM, global_step)
            
            if (global_step%500==0):
                print(">>Global step:",global_step,"Saving the model..")
                agent.save_restore_Model(restore=False, epoch=global_step)

            global_step+=1
            startB=endB


def showGridCells(agent):
    from ratSimulator import RatSimulator
    timesteps=1000000
    env=RatSimulator(timesteps)
    print(">>Generating trajectory")
    vel, angVel, pos, angle =env.generateTrajectory()

    inputData=np.zeros((timesteps,3))
    inputData[:,0]=vel
    inputData[:,1]=np.sin(angVel)
    inputData[:,2]=np.cos(angVel)

    init_LSTMState=np.zeros((PlaceCells_units + HeadCells_units))
    init_LSTMState[:PlaceCells_units]=dataGenerator.computePlaceCellsDistrib(np.reshape(pos[0], (1,2)), place_cell_centers)
    init_LSTMState[PlaceCells_units:]=dataGenerator.computeHeadCellsDistrib(np.reshape(angle[0], (1,1)), head_cell_centers)

    print(">>Computing Activity Map..")
    agent.showGridCells(inputData, init_LSTMState, pos, timesteps)

if __name__ == '__main__':
    parser=argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("mode", help="generate - will generate trajectories and train the model \ntrain - will train the model \nshowcells - will create the activity maps of the neurons")
    args=parser.parse_args()
    with tf.Session() as sess:
        try:
            agent=network(sess, lr=learning_rate, hu=LSTMUnits, lu=linearLayerUnits, place_units=PlaceCells_units, head_units=HeadCells_units, clipping=clipping, weightDecay=weightDecay, batch_size=batch_size, num_features=num_features, n_steps=numberSteps)

            if(args.mode=="generate" or args.mode=="train"):
                if(args.mode=="generate"):
                    #If generating new data, remove the old files 
                    dataGenerator.generateData()

                if os.path.exists("agentBackup"):
                    print("Loading the model..")
                    agent.save_restore_Model(restore=True)
                    global_step=sess.run(agent.epoch)
                
                trainAgent(agent)
            elif(args.mode=="showcells"):
                agent.save_restore_Model(restore=True)
                showGridCells(agent)
                
        except (KeyboardInterrupt,SystemExit):
            print("\n\nProgram shut down, saving the model..")
            agent.save_restore_Model(restore=False, epoch=global_step)
            print("\n\nModel saved!\n\n")
            raise