import numpy as np 
import matplotlib.pyplot as plt
class RatSimulator():
    def __init__(self, n_steps):
        self.number_steps=n_steps
        self.dt=0.02
        self.maxGap=2.17
        self.minGap=0.03

        self.velScale=0.13
        self.mAngVel=0
        self.stddevAngVel=330


    def generateTrajectory(self):
        velocities=np.zeros((self.number_steps))
        angVel=np.zeros((self.number_steps))
        positions=np.zeros((self.number_steps, 2))
        angle=np.zeros((self.number_steps))

        for t in range(self.number_steps):
            #Initialize the agent randomly in the environment
            if(t==0):
                pos=np.random.uniform(low=0, high=2.2, size=(2))
                facAng=np.random.uniform(low=-np.pi, high=np.pi)
                prevVel=0

            #Check if the agent is near a wall
            if(self.checkWallAngle(facAng, pos)):
                #if True, compute in which direction turning by 90 deg
                rotVel=np.deg2rad(np.random.normal(self.mAngVel, self.stddevAngVel))
                dAngle=self.computeRot(facAng, pos) + rotVel*0.02
                #Velocity reduction factor
                vel=np.squeeze(prevVel - (prevVel*0.25))
            #If the agent is not near a wall, randomly sampling velocity and angVelocity
            else:
                #Sampling velocity
                vel=np.random.rayleigh(self.velScale)
                #Sampling angular velocity
                rotVel=np.deg2rad(np.random.normal(self.mAngVel, self.stddevAngVel))
                dAngle=rotVel*0.02

            #Update the position of the agent
            newPos=pos + (np.asarray([np.cos(facAng), np.sin(facAng)])*vel)*self.dt
            
            #Update the facing angle of the agent
            newFacAng=(facAng + dAngle)
            #Keep the orientation between -np.pi and np.pi
            if(np.abs(newFacAng)>=(np.pi)):     
                newFacAng=-1*np.sign(newFacAng)*(np.pi - (np.abs(newFacAng)- np.pi))

            velocities[t]=vel
            angVel[t]=rotVel
            positions[t]=pos
            angle[t]=facAng
            
            pos=newPos
            facAng=newFacAng
            prevVel=vel
        
        '''
        #USED TO DISPLAY THE TRAJECTORY ONCE FINISHED
        fig=plt.figure(figsize=(15,15))
        ax=fig.add_subplot(111)
        ax.set_title("Trajectory agent")
        ax.plot(positions[:,0], positions[:,1])
        ax.set_xlim(0,2.2)
        ax.set_ylim(0,2.2)

        plt.show()
        '''
        
        return velocities, angVel, positions, angle

    #HELPING FUNCTIONS
    def checkWallAngle(self, ratAng, pos):
        #print("Rat orientation:", ratAng)
        if((0<=ratAng and ratAng<=(np.pi/2)) and np.any(pos>self.maxGap)):
          return True
        elif((ratAng>=(np.pi/2) and ratAng<=np.pi) and (pos[0]<self.minGap or pos[1]>self.maxGap)):
          return True
        elif((ratAng>=-np.pi and ratAng<=(-np.pi/2)) and np.any(pos<self.minGap)):
          return True
        elif((ratAng>=(-np.pi/2) and ratAng<=0) and (pos[0]>self.maxGap or pos[1]<self.minGap)):
          return True
        else:
          return False
    
    def computeRot(self,ang, pos):
        rot=0
        if(ang>=0 and ang<=(np.pi/2)):
          if(pos[1]>self.maxGap):
            rot=-ang
          elif(pos[0]>self.maxGap):
            rot=np.pi/2-ang
        elif(ang>=(np.pi/2) and ang<=np.pi):
          if(pos[1]>self.maxGap):
            rot=np.pi-ang
          elif(pos[0]<self.minGap):
            rot=np.pi/2 -ang
        elif(ang>=-np.pi and ang<=(-np.pi/2)):
          if(pos[1]<self.minGap):
            rot=-np.pi - ang
          elif(pos[0]<self.minGap):
            rot=-(ang + np.pi/2)
        else:
          if(pos[1]<self.minGap):
            rot=-ang
          elif(pos[0]>self.maxGap):
            rot=(-np.pi/2) - ang

        return rot