import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
class Trainer():
    def __init__(self, agent, pcu, numSteps):
        self.agent=agent
        self.PlaceCells_units=pcu
        self.numberSteps=numSteps

    def training(self, X, Y, init_X, epoch):
        '''
        #Stores the initializer weights to restore at the end of each epoch training.
        initializers=[]
        '''

        #Store the LSTM_state at each timestep. Use these instead of initialize new ones 
        #except at timestep=0
        hidden_state=np.zeros((10, 128))
        cell_state=np.zeros((10, 128))

        #Stores the means of the losses among a training epoch.
        #Used to show the stats on tensorboard
        mn_loss=0

        #Divide the sequence in 100 steps in order to apply TBTT of 100 timesteps.
        for startB in range(0, self.numberSteps, 100):
            endB=startB+100

            #Retrieve the inputs for the 100 timesteps
            xBatch=X[:,startB:endB]

            #Retrieve the labels for the 100 timesteps
            yBatchPlaceCells=Y[:, startB:endB, : self.PlaceCells_units]
            yBatchHeadCells=Y[:, startB:endB, self.PlaceCells_units : ]

            #When the timestep=0, initialize the hidden and cell state of LSTm using init_X. if not timestep=0, the network will use cell_state and hidden_state
            feed_dict={ self.agent.X: xBatch, 
                        self.agent.LabelPlaceCells: yBatchPlaceCells,
                        self.agent.LabelHeadCells: yBatchHeadCells,
                        self.agent.placeCellGround: init_X[:, :self.PlaceCells_units], 
                        self.agent.headCellGround: init_X[:, self.PlaceCells_units:],
                        self.agent.timestep: startB,
                        self.agent.old_cell_state: cell_state,
                        self.agent.old_hidden_state: hidden_state,
                        self.agent.keepProb: 0.5}
            
            _, meanLoss, HeadLoss, PlaceLoss, lstm_state=self.agent.sess.run([self.agent.opt,
                                                                        self.agent.meanLoss,
                                                                        self.agent.errorHeadCells,
                                                                        self.agent.errorPlaceCells,
                                                                        self.agent.hidden_cell_statesTuple], feed_dict=feed_dict)

            #We want that for the next batch of 100 timesteps, the hidden state and cell state of the LSTM cells 
            #have the same values of the h_state and c_state oututed at the previous timestep training
            hidden_state=lstm_state[0]
            cell_state=lstm_state[1]

            '''
            #At each training epoch, after the training for timestep=0 save the values
            #of the weights used to initialize the hidden and cell state. 
            #We are going to use them at the timestep=0 of the next training epoch
            if startB==0:#it means after the first training
                initializers.append(self.agent.sess.run(self.agent.Wcp))
                initializers.append(self.agent.sess.run(self.agent.Wcd))
                initializers.append(self.agent.sess.run(self.agent.Whp))
                initializers.append(self.agent.sess.run(self.agent.Whd))
            elif startB==self.numberSteps-100:
                #At the end of each training epoch, set the values of the weights initializers
                #as when they were after timestep=0. They have changed during the other 749 timesteps.
                self.agent.sess.run(tf.assign(self.agent.Wcp, initializers[0]))
                self.agent.sess.run(tf.assign(self.agent.Wcd, initializers[1]))
                self.agent.sess.run(tf.assign(self.agent.Whp, initializers[2]))
                self.agent.sess.run(tf.assign(self.agent.Whd, initializers[3]))
            '''

            mn_loss += meanLoss/(self.numberSteps//100)

        #training epoch finished, save the errors for tensorboard
        mergedData=self.agent.sess.run(self.agent.mergeEpisodeData, feed_dict={self.agent.mn_loss: mn_loss})
        
        self.agent.file.add_summary(mergedData, epoch)

    def testing(self, X, init_X, positions_array, pcc, epoch):
        #Store the LSTM_state at each timestep. Use these instead of initialize new ones 
        #except at timestep=0
        hidden_state=np.zeros((10, 128))
        cell_state=np.zeros((10, 128))

        avgDistance=0

        displayPredTrajectories=np.zeros((10,800,2))

        #Divide the sequence in 100 steps
        for startB in range(0, self.numberSteps, 100):
            endB=startB+100

            #Retrieve the inputs for the timestep
            xBatch=X[:,startB:endB]

            #When the timestep=0, initialize the hidden and cell state of LSTm using init_X. if not timestep=0, the network will use cell_state and hidden_state
            feed_dict={ self.agent.X: xBatch, 
                        self.agent.placeCellGround: init_X[:, :self.PlaceCells_units], 
                        self.agent.headCellGround: init_X[:, self.PlaceCells_units:],
                        self.agent.timestep: startB,
                        self.agent.old_cell_state: cell_state,
                        self.agent.old_hidden_state: hidden_state,
                        self.agent.keepProb: 1}
            
            lstm_state, placeCellLayer=self.agent.sess.run([self.agent.hidden_cell_statesTuple, self.agent.OutputPlaceCellsLayer], feed_dict=feed_dict)
            
            #We want that for the next timestep training the hidden state and cell state of the LSTM cells 
            #have the same values of the h_state and c_state outputed at the previous timestep training
            hidden_state=lstm_state[0]
            cell_state=lstm_state[1]    
            
            #retrieve the position in these 100 timesteps
            positions=positions_array[:,startB:endB]
            #Retrieve which cell has been activated. Placecell has shape 1000,256. idx has shape 1000,1
            idx=np.argmax(placeCellLayer, axis=1)
            
            #Retrieve the place cell center of the activated place cell
            predPositions=pcc[idx]

            #Update the predictedTrajectory.png
            if epoch%8000==0:
                displayPredTrajectories[:,startB:endB]=np.reshape(predPositions,(10,100,2))

            #Compute the distance between truth position and place cell center
            distances=np.sqrt(np.sum((predPositions - np.reshape(positions, (-1,2)))**2, axis=1))
            avgDistance +=np.mean(distances)/(self.numberSteps//100)
        
        #testing epoch finished, save the accuracy for tensorboard
        mergedData=self.agent.sess.run(self.agent.mergeAccuracyData, feed_dict={self.agent.avgD: avgDistance})
        
        self.agent.file.add_summary(mergedData, epoch)

        #Compare predicted trajectory with real trajectory
        if epoch%8000==0:
            rows=3
            cols=3
            fig=plt.figure(figsize=(40, 40))
            for i in range(rows*cols):
                ax=fig.add_subplot(rows, cols, i+1)
                #plot real trajectory
                plt.plot(positions_array[i,:,0], positions_array[i,:,1], 'b', label="Truth Path")
                #plot predicted trajectory
                plt.plot(displayPredTrajectories[i,:,0], displayPredTrajectories[i,:,1], 'g', label="Predicted Path")
                plt.legend()
                ax.set_xlim(0,2.2)
                ax.set_ylim(0,2.2)

            fig.savefig('predictedTrajectory.png')