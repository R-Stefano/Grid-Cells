import numpy as np
import matplotlib.pyplot as plt
import pickle
from ratSimulator import RatSimulator

class dataGenerator():
    def __init__(self, batch_size, number_steps, num_features, pc_units, hd_units):
        #HYPERPARAMETERS
        self.batch_size=batch_size
        self.number_steps=number_steps
        self.num_features=num_features
        self.placeCell_units=pc_units
        self.headCell_units=hd_units

        self.ratSimulator=RatSimulator(self.number_steps)

        
    def generateData(self):
        inputData=np.zeros((self.batch_size+10, self.number_steps,3))

        velocities=np.zeros((self.batch_size+10,self.number_steps))
        angVelocities=np.zeros((self.batch_size+10,self.number_steps))
        angles=np.zeros((self.batch_size+10,self.number_steps))
        positions=np.zeros((self.batch_size+10, self.number_steps,2))

        print(">>Generating trajectories")
        for i in range(self.batch_size+10):
            vel, angVel, pos, angle=self.ratSimulator.generateTrajectory()    

            velocities[i]=vel
            angVelocities[i]=angVel
            angles[i]=angle
            positions[i]=pos
        for t in range(self.number_steps):
            inputData[:,t,0],inputData[:,t,1],inputData[:,t,2]=velocities[:,t], np.sin(angVelocities[:,t]), np.cos(angVelocities[:,t])

        mydict={"X":inputData[:-10],"pos":positions[:-10],"angle":angles[:-10]}
        with open('trajectoriesData.pickle', 'wb') as f:
            pickle.dump(mydict, f)
        
        mydict={"X":inputData[-10:],"pos":positions[-10:], "angle":angles[-10:]}
        with open('trajectoriesDataTesting.pickle', 'wb') as f:
            pickle.dump(mydict, f)
        

    def computePlaceCellsDistrib(self, positions, cellCenters):
        num_cells=cellCenters.shape[0]
        batch_size=positions.shape[0]
        #Place Cell scale
        sigma=0.3#0.01 NOT 0.01 PAPER ERROR

        summs=np.zeros(batch_size)
        #Store [envs,256] elements. Every row stores the distribution for a trajectory
        distributions=np.zeros((batch_size,num_cells))
        #We have 256 elements in the Place Cell Distribution. For each of them
        for i in range(num_cells):
            #positions has shape [envs,2] and cellCenters[i] has shape [envs,2]
            l2Norms=np.sum((positions - cellCenters[i])**2, axis=1)

            #placeCells has shape [envs,1]
            placeCells=np.exp(-(l2Norms/(2*sigma**2)))

            distributions[:,i]=placeCells
            summs +=placeCells
        distributions=distributions/summs[:,None]
        return distributions

    def computeHeadCellsDistrib(self,facingAngles, cellCenters):
        num_cells=cellCenters.shape[0]
        batch_size=facingAngles.shape[0]
        #Concentration parameter
        k=20

        summs=np.zeros(batch_size)
        #Store [envs,12] elements. Every row stores the distribution for a trajectory
        distributions=np.zeros((batch_size,num_cells))
        #We have 12 elements in the Head Direction Cell Distribution. For each of them
        for i in range(num_cells):
            #facingAngles has shape [envs, 1] while cellCenters[i] has shape [envs,1]
            headDirects=np.squeeze(np.exp(k*np.cos(facingAngles - cellCenters[i])))
            distributions[:,i]=headDirects
            summs+=headDirects
        
        distributions=distributions/summs[:,None]

        return distributions
