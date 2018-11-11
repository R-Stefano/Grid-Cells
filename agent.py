import tensorflow as tf 

#Define the agent structure network
class Network():
    def __init__(self, session, lr, hu, lu, place_units, head_units, clipping, weightDecay, batch_size, num_features, n_steps):
        self.sess=session
        self.epoch=tf.Variable(0, trainable=False)
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

        self.keepProb=tf.placeholder(tf.float32, name="keep_prob")

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

            self.linearLayer=tf.matmul(self.reshapedOut, self.W1) + self.B1
            
            #Compute Linear layer and apply dropout
            self.linearLayerDrop=tf.nn.dropout(self.linearLayer, self.keepProb)

        with tf.variable_scope("Place_Cells_Units"):
            self.W2=tf.get_variable("Weights_LinearDecoder_placeCells", [self.LinearLayer_units, self.PlaceCells_units], initializer=tf.contrib.layers.xavier_initializer(), 
                                                                                                                            regularizer=tf.contrib.layers.l2_regularizer(self.weight_decay))
            self.B2=tf.get_variable("Biases_LinearDecoder_placeCells", [self.PlaceCells_units], initializer=tf.contrib.layers.xavier_initializer())
            
            #Compute the predicted Place Cells Distribution
            self.OutputPlaceCellsLayer=tf.nn.softmax(tf.matmul(self.linearLayerDrop, self.W2) + self.B2, name="Output_Place_Cells")  

        with tf.variable_scope("Head_Cells_Units"):
            self.W3=tf.get_variable("Weights_LinearDecoder_HeadDirectionCells", [self.LinearLayer_units, self.HeadCells_units], initializer=tf.contrib.layers.xavier_initializer(), 
                                                                                                                                regularizer=tf.contrib.layers.l2_regularizer(self.weight_decay))
            self.B3=tf.get_variable("Biases_LinearDecoder_HeadDirectionCells", [self.HeadCells_units], initializer=tf.contrib.layers.xavier_initializer())   
            
            #Compute the predicted Head-direction Cells Distribution
            self.OutputHeadCellsLayer=tf.nn.softmax(tf.matmul(self.linearLayerDrop, self.W3) + self.B3, name="Output_Head_Cells")
  
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