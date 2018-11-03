import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.signal import correlate2d
#Define the agent structure network
class network():
    def __init__(self, session, lr, hu, lu, place_units, head_units, clipping, weightDecay, batch_size, num_features, n_steps):
        self.sess=session
        self.epoch=tf.Variable(0, trainable=False)
        self.numberSteps=n_steps

        self.bins=32
        self.factor=2.2/self.bins
        self.activityMap=np.zeros((lu, self.bins, self.bins))
        self.counterActivityMap=np.zeros((lu, self.bins, self.bins))
        #HYPERPARAMETERS
        self.learning_rate=lr
        self.Hidden_units=hu
        self.LinearLayer_units=lu
        self.PlaceCells_units=place_units
        self.HeadCells_units=head_units
        self.clipping=clipping
        self.weight_decay=tf.constant(weightDecay, dtype=tf.float32)
        self.batch_size=batch_size
        self.num_features=num_features

        self.buildNetwork()
        self.buildTraining()
        self.buildTensorBoardStats()

        self.sess.run(tf.global_variables_initializer())

        self.saver=tf.train.Saver()
        self.file=tf.summary.FileWriter("tensorboard/", self.sess.graph)

    def buildNetwork(self):
        self.X=tf.placeholder(tf.float32, shape=[None, 100, self.num_features], name="input")

        self.placeCellGround=tf.placeholder(tf.float32, shape=[None, self.PlaceCells_units], name="Groud_Truth_Place_Cell")
        self.headCellGround=tf.placeholder(tf.float32, shape=[None, self.HeadCells_units], name="Groud_Truth_Head_Cell")

        self.timestep=tf.placeholder(tf.int32, name="timestep")

        self.old_cell_state=tf.placeholder(tf.float32, name="old_cell_state")
        self.old_hidden_state=tf.placeholder(tf.float32, name="old_hidden_state")

        with tf.variable_scope("LSTM_initialization"):
            #Initialize the Hidden state and Cell state of the LSTM unit using feeding the Ground Truth Distributio at timestep 0. Both have size [batch_size, Hidden_units]
            self.Wcp=tf.get_variable("Initial_state_cp", [self.PlaceCells_units,self.Hidden_units], initializer=tf.contrib.layers.xavier_initializer())
            self.Wcd=tf.get_variable("Initial_state_cd", [self.HeadCells_units,self.Hidden_units],  initializer=tf.contrib.layers.xavier_initializer())
            self.Whp=tf.get_variable("Hidden_state_hp",  [self.PlaceCells_units,self.Hidden_units], initializer=tf.contrib.layers.xavier_initializer())
            self.Whd=tf.get_variable("Hidden_state_hd",  [self.HeadCells_units,self.Hidden_units],  initializer=tf.contrib.layers.xavier_initializer())

            #Compute self.hidden_state 
            self.hidden_state= tf.matmul(self.placeCellGround, self.Wcp) + tf.matmul( self.headCellGround, self.Wcd)
            #Compute self.cell_state
            self.cell_state=tf.matmul(self.placeCellGround, self.Whp) + tf.matmul( self.headCellGround, self.Whd)

            #Store self.cell_state and self.hidden_state tensors as elements of a single list.
            #If is going to be timestep=0, initialize the hidden and cell state using the Ground Truth Distributions. 
            #Otherwise, use the hidden state and cell state from the previous timestep passed using the placeholders        
            self.LSTM_state=tf.cond(tf.equal(self.timestep,0), 
                                    lambda: tf.nn.rnn_cell.LSTMStateTuple(self.hidden_state, self.cell_state), 
                                    lambda: tf.nn.rnn_cell.LSTMStateTuple(self.old_hidden_state, self.old_cell_state))

        with tf.variable_scope("LSTM"):
            #Define the single LSTM cell with the number of hidden units
            self.lstm_cell=tf.contrib.rnn.LSTMCell(self.Hidden_units, name="LSTM_Cell")

            #Feed an input of shape [batch_size, 100, features]
            #self.output is a tensor of shape [batch_size, 100, hidden_units]
            #out_hidden_statesTuple is a list of 2 elements: self.cell_state, self.hidden_state where self.output[:, -1, :]=self.cell_state
            self.output, self.hidden_cell_statesTuple=tf.nn.dynamic_rnn(cell=self.lstm_cell, inputs=self.X, initial_state=self.LSTM_state)

        with tf.variable_scope("Linear_Decoder"):
            self.W1=tf.get_variable("Weights_LSTM_LinearDecoder", [self.Hidden_units, self.LinearLayer_units], initializer=tf.contrib.layers.xavier_initializer())
            self.B1=tf.get_variable("Biases_LSTM_LinearDecoder", [self.LinearLayer_units], initializer=tf.contrib.layers.xavier_initializer())
            
            #we can't feed a tensor of shape [10,100,128] to the linear layer. We treat each timestep in every trajectory as an example
            #we now have a matrix of shape [10*100,128] which can be fed to the linear layer. The result is the same as 
            #looping 100 times through each timestep examples.
            self.reshapedOut=tf.reshape(self.output, (-1, self.Hidden_units))
            
            #Compute Linear layer and apply dropout
            self.linearLayer=tf.nn.dropout(tf.matmul(self.reshapedOut, self.W1) + self.B1, 0.5)

        with tf.variable_scope("Place_Cells_Units"):
            self.W2=tf.get_variable("Weights_LinearDecoder_placeCells", [self.LinearLayer_units, self.PlaceCells_units], initializer=tf.contrib.layers.xavier_initializer(), 
                                                                                                                            regularizer=tf.contrib.layers.l2_regularizer(self.weight_decay))
            self.B2=tf.get_variable("Biases_LinearDecoder_placeCells", [self.PlaceCells_units], initializer=tf.contrib.layers.xavier_initializer())
            
            #Compute the predicted Place Cells Distribution
            self.OutputPlaceCellsLayer=tf.nn.softmax(tf.matmul(self.linearLayer, self.W2) + self.B2, name="Output_Place_Cells")  

        with tf.variable_scope("Head_Cells_Units"):
            self.W3=tf.get_variable("Weights_LinearDecoder_HeadDirectionCells", [self.LinearLayer_units, self.HeadCells_units], initializer=tf.contrib.layers.xavier_initializer(), 
                                                                                                                                regularizer=tf.contrib.layers.l2_regularizer(self.weight_decay))
            self.B3=tf.get_variable("Biases_LinearDecoder_HeadDirectionCells", [self.HeadCells_units], initializer=tf.contrib.layers.xavier_initializer())   
            
            #Compute the predicted Head-direction Cells Distribution
            self.OutputHeadCellsLayer=tf.nn.softmax(tf.matmul(self.linearLayer, self.W3) + self.B3, name="Output_Head_Cells")
  
    def buildTraining(self):
        #Fed the Ground Truth Place Cells Distribution and Head Direction Cells Distribution
        self.LabelPlaceCells=tf.placeholder(tf.float32, shape=[None, 100, self.PlaceCells_units], name="Labels_Place_Cells")
        self.LabelHeadCells=tf.placeholder(tf.float32,  shape=[None, 100, self.HeadCells_units], name="Labels_Head_Cells")
        
        #Compute the errors for each neuron in each trajectory for each timestep [1000,256] and [1000,12] errors
        self.errorPlaceCells= - tf.reduce_sum(tf.reshape(self.LabelPlaceCells, (-1, self.PlaceCells_units)) * tf.log(self.OutputPlaceCellsLayer + 1e-9), axis=1, name="Error_PlaceCells")
        self.errorHeadCells= - tf.reduce_sum(tf.reshape(self.LabelHeadCells, (-1, self.HeadCells_units)) * tf.log(self.OutputHeadCellsLayer + 1e-9), axis=1, name="Error_HeadCells")
        
        #Convert back the tensor from [1000, 1] to [10,100]
        self.reshapedErrors=tf.reshape((self.errorPlaceCells + self.errorHeadCells), (-1,100))
        #Compute the truncated backprop error for each trajectory (SUMMING THE ERRORS). [10,100] -> [10,1]
        self.truncErrors=tf.reduce_sum(self.reshapedErrors, axis=1)
        #Compute mean among truncated errors [10,1] -> [1] (mean error)
        self.meanLoss=tf.reduce_mean(self.truncErrors, name="mean_error")
        
        self.optimizer=tf.train.RMSPropOptimizer(self.learning_rate, momentum=0.9)

        self.gvs=self.optimizer.compute_gradients(self.meanLoss)

        #Apply gradient clipping to parameters: Place Cells units (weights, biases) , Head Cells units (weights, biases)
        self.gvs[-4]=[tf.clip_by_value(self.gvs[-4][0], -self.clipping, self.clipping), self.gvs[-4][1]]
        self.gvs[-3]=[tf.clip_by_value(self.gvs[-3][0], -self.clipping, self.clipping), self.gvs[-3][1]]
        self.gvs[-2]=[tf.clip_by_value(self.gvs[-2][0], -self.clipping, self.clipping), self.gvs[-2][1]]
        self.gvs[-1]=[tf.clip_by_value(self.gvs[-1][0], -self.clipping, self.clipping), self.gvs[-1][1]]

        self.opt=self.optimizer.apply_gradients(self.gvs)
    
    def buildTensorBoardStats(self):
        #Episode data
        self.mn_loss=tf.placeholder(tf.float32)
        self.mergeEpisodeData=tf.summary.merge([tf.summary.scalar("mean_loss", self.mn_loss)])

        self.avgD=tf.placeholder(tf.float32)
        self.mergeAccuracyData=tf.summary.merge([tf.summary.scalar("average_distance", self.avgD)])
    
    def save_restore_Model(self, restore, epoch=0):
        if restore:
            self.saver.restore(self.sess, "agentBackup/graph.ckpt")
        else:
            self.sess.run(self.epoch.assign(epoch))
            self.saver.save(self.sess, "agentBackup/graph.ckpt")
    
    def training(self, X, Y, init_X, epoch):
        #Stores the initializer weights to restore at the end of each epoch training.
        initializers=[]

        #Store the LSTM_state at each timestep. Use these instead of initialize new ones 
        #except at timestep=0
        hidden_state=np.zeros((self.batch_size, self.Hidden_units))
        cell_state=np.zeros((self.batch_size, self.Hidden_units))

        #Stores the means of the losses among a training epoch.
        #Used to show the stats on tensorboard
        mn_loss=0

        #Divide the sequence in 100 steps in order to apply TBTT of 100 timesteps.
        batches=int(self.numberSteps//100)
        lastBatch=batches-1
        for b in range(batches):
            startB=0
            endB=startB+100

            #Retrieve the labels for the 100 timesteps
            yBatchPlaceCells=Y[:, startB:endB, : self.PlaceCells_units]
            yBatchHeadCells=Y[:, startB:endB, self.PlaceCells_units : ]

            #Retrieve the inputs for the 100 timesteps
            xBatch=X[:,startB:endB]

            #When the timestep=0, initialize the hidden and cell state of LSTm using init_X. if not timestep=0, the network will use cell_state and hidden_state
            feed_dict={ self.X: xBatch, 
                        self.LabelPlaceCells: yBatchPlaceCells,
                        self.LabelHeadCells: yBatchHeadCells,
                        self.placeCellGround: init_X[:, :self.PlaceCells_units], 
                        self.headCellGround: init_X[:, self.PlaceCells_units:],
                        self.timestep: startB,
                        self.old_cell_state: cell_state,
                        self.old_hidden_state: hidden_state}
            
            _, meanLoss, HeadLoss, PlaceLoss, lstm_state=self.sess.run([self.opt,
                                                                        self.meanLoss,
                                                                        self.errorHeadCells,
                                                                        self.errorPlaceCells,
                                                                        self.hidden_cell_statesTuple], feed_dict=feed_dict)

            #We want that for the next batch of 100 timesteps, the hidden state and cell state of the LSTM cells 
            #have the same values of the h_state and c_state oututed at the previous timestep training
            hidden_state=lstm_state[0]
            cell_state=lstm_state[1]

            #At each training epoch, after the training for timestep=0 save the values
            #of the weights used to initialize the hidden and cell state. 
            #We are going to use them at the timestep=0 of the next training epoch
            if b==0:#it means after the first training
                initializers.append(self.sess.run(self.Wcp))
                initializers.append(self.sess.run(self.Wcd))
                initializers.append(self.sess.run(self.Whp))
                initializers.append(self.sess.run(self.Whd))
            elif b==lastBatch:
                #At the end of each training epoch, set the values of the weights initializers
                #as when they were after timestep=0. They have changed during the other 749 timesteps.
                self.sess.run(tf.assign(self.Wcp, initializers[0]))
                self.sess.run(tf.assign(self.Wcd, initializers[1]))
                self.sess.run(tf.assign(self.Whp, initializers[2]))
                self.sess.run(tf.assign(self.Whd, initializers[3]))

            mn_loss += meanLoss/batches

        #training epoch finished, save the errors for tensorboard
        mergedData=self.sess.run(self.mergeEpisodeData, feed_dict={self.mn_loss: mn_loss})
        
        self.file.add_summary(mergedData, epoch)

        startB=endB

    def testing(self, X, init_X, positions_array, pcc, epoch):
        #Store the LSTM_state at each timestep. Use these instead of initialize new ones 
        #except at timestep=0
        hidden_state=np.zeros((100, self.Hidden_units))
        cell_state=np.zeros((100, self.Hidden_units))

        avgD=0

        #Divide the sequence in 100 steps in order to apply TBTT of 100 timesteps.
        batches=int(self.numberSteps//100)
        startB=0
        for b in range(batches):
            endB=startB+100

            #Retrieve the inputs for the timestep
            xBatch=X[:,startB:endB]

            #When the timestep=0, initialize the hidden and cell state of LSTm using init_X. if not timestep=0, the network will use cell_state and hidden_state
            feed_dict={ self.X: xBatch, 
                        self.placeCellGround: init_X[:, :self.PlaceCells_units], 
                        self.headCellGround: init_X[:, self.PlaceCells_units:],
                        self.timestep: startB,
                        self.old_cell_state: cell_state,
                        self.old_hidden_state: hidden_state}
            
            lstm_state, placeCellLayer=self.sess.run([self.hidden_cell_statesTuple, self.OutputPlaceCellsLayer], feed_dict=feed_dict)
            

            #We want that for the next timestep training the hidden state and cell state of the LSTM cells 
            #have the same values of the h_state and c_state oututed at the previous timestep training
            hidden_state=lstm_state[0]
            cell_state=lstm_state[1]    
            
            #retrieve the position in these 100 timesteps
            positions=positions_array[:,startB:endB]
            #placecell is in shape 1000,256. idx has shape 1000,1
            idx=np.argmax(placeCellLayer, axis=1)
            
            #it has shape [1000,2]
            predPositions=pcc[idx]

            distances=np.sqrt(np.sum((predPositions - np.reshape(positions, (-1,2)))**2, axis=1))
            avgD +=np.mean(distances)/batches
        
        #testing epoch finished, save the accuracy for tensorboard
        mergedData=self.sess.run(self.mergeAccuracyData, feed_dict={self.avgD: avgD})
        
        self.file.add_summary(mergedData, epoch)

        '''
        #for every batch
        for i in range(10):
            #for every timestep
            for j in range(100):
                #retrieve most active neuron
                idx=np.argmax(placeCellLayerReshaped[i,j])
                print(pcc[idx])
                #store in the array
                predPositions[i,j+startB-1]=pcc[idx]

        rows=3
        cols=3
        fig=plt.figure(figsize=(40, 40))
        for i in range(1, rows*cols+1):
            fig.add_subplot(rows, cols, i)
            #plot real trajectory
            plt.plot(positions_array[i,:,0], positions_array[i,:,1], 'b')
            #plot predicted trajectory
            plt.plot(predPositions[i,:,0], predPositions[i,:,1], 'r')
            plt.axis('off')

        fig.savefig('predictedTrajectory.jpg')
        '''


    
    def showGridCells(self, X, init_X, positions_array):
        #Feed 1k examples at time to avoid memory problems. Otherwise (10000*100=1million matrix)
        start=0
        for i in range(10):
            end=start+1000
            #Store the LSTM_state at each timestep. Use these instead of initialize new ones 
            #except at timestep=0
            hidden_state=np.zeros((100, self.Hidden_units))
            cell_state=np.zeros((100, self.Hidden_units))

            #Divide the sequence in 100 steps in order to apply TBTT of 100 timesteps.
            batches=int(self.numberSteps//100)
            startB=0
            for b in range(batches):
                endB=startB+100

                #Retrieve the inputs for the timestep
                xBatch=X[start:end,startB:endB]

                #When the timestep=0, initialize the hidden and cell state of LSTm using init_X. if not timestep=0, the network will use cell_state and hidden_state
                feed_dict={ self.X: xBatch, 
                            self.placeCellGround: init_X[start:end, :self.PlaceCells_units], 
                            self.headCellGround: init_X[start:end, self.PlaceCells_units:],
                            self.timestep: startB,
                            self.old_cell_state: cell_state,
                            self.old_hidden_state: hidden_state}
                
                lstm_state, linearNeurons=self.sess.run([self.hidden_cell_statesTuple, self.linearLayer], feed_dict=feed_dict)
                

                #We want that for the next timestep training the hidden state and cell state of the LSTM cells 
                #have the same values of the h_state and c_state oututed at the previous timestep training
                hidden_state=lstm_state[0]
                cell_state=lstm_state[1]

                positions=np.reshape(positions_array[start:end,startB:endB],(-1,2))

                #save the value of the neurons in the linear layer at each timestep
                for t in range(linearNeurons.shape[0]):
                    #Compute which bins are for each position
                    bin_x, bin_y=(positions[t]//self.factor).astype(int)

                    if(bin_y==self.bins):
                        bin_y=self.bins-1
                    elif(bin_x==self.bins):
                        bin_x=self.bins-1

                    #Now there are the 512 values of the same location
                    self.activityMap[:,bin_y, bin_x]+=linearNeurons[t]#linearNeurons must be a vector of 512
                    self.counterActivityMap[:, bin_y, bin_x]+=np.ones((512))

                startB=endB
            
            start=end

        self.counterActivityMap[self.counterActivityMap==0]=1
        #Compute average value
        result=self.activityMap/self.counterActivityMap

        os.makedirs("activityMaps", exist_ok=True)
        os.makedirs("corrMaps", exist_ok=True)


        '''
        I want to show 64 neurons in each image so 8x8
        it means that there will be 8 images
        '''
        cols=16
        rows=32
        count=0
        #Save images
        fig=plt.figure(figsize=(40, 40))
        for i in range(1, rows*cols+1):
            fig.add_subplot(rows, cols, i)
            plt.imshow(2*((result[count]-np.min(result[count]))/(np.max(result[count])-np.min(result[count])))-1, cmap="jet", origin="lower")#, interpolation="gaussian")
            plt.axis('off')

            count+=1
        fig.subplots_adjust(wspace=0.1, hspace=0.1)        
        fig.savefig('activityMaps/neurons.jpg')

        count=0
        fig=plt.figure(figsize=(40, 40))
        for i in range(1, rows*cols+1):
            fig.add_subplot(rows, cols, i)
            normMap=2*((result[count]-np.min(result[count]))/(np.max(result[count])-np.min(result[count])))-1
            plt.imshow(correlate2d(normMap, normMap), cmap="jet", origin="lower")
            plt.axis('off')

            count+=1
        fig.subplots_adjust(wspace=0.1, hspace=0.1)
        fig.savefig('corrMaps/neurons.jpg')

        #Reset the maps
        self.activityMap=np.zeros((self.LinearLayer_units, self.bins, self.bins))
        self.counterActivityMap=np.zeros((self.LinearLayer_units, self.bins, self.bins))



