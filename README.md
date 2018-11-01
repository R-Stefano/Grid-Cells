# Vector-based Navigation using Grid-like Representations in Artificial Agents

![Heading Image](http://demiledge.com/structureFiles/Images/VBN0.jpg)

## About
Replicating Google Deepmind's paper ["Vector-based Navigation using Grid-like Representations in Artificial Agents"](https://deepmind.com/blog/grid-cells/).

It has been a long day, you are really tired and you just arrived home. You open the door and you don't have to turn on the lights but you can successfully reach the bed and felt asleep. 
How it has been possible?

You were tired and you didn't even see where you were going. Despite this, you moved through a familiar environment easily. 

Mainly, this achievement has been possible due to the work done in the Hippocampus and in the Entorhinal cortex. In fact, it is believed that particular neurons in these regions of the brain allow us to self-localize and to navigate through environments. In these regions of the brain there are neurons that fire depending on where we are: Place cells and others that fire based on which direction we are facing: Head-direction Cells..

[What are Grid cells?  Full article about the paper here]() (add link)

## Dependencies
* Tensorflow
* Numpy
* Matplotlib

## Network
![network Image](http://demiledge.com/structureFiles/Images/VBN5.jpg)

## Getting started
`ratSimulator.py` contains the code used to generate the trajectories. The simulator is based on [this paper](https://journals.plos.org/ploscompbiol/article/file?id=10.1371/journal.pcbi.1002553&type=printable).

`dataGenerator.py` is used to create the **Place Cell Distributions** and **Head Cell Distributions**

`agent.py` contains the architecture of the network in Tensorflow.

In order to start the training, `main.py` must be called.

```
	python3 main.py generate
```

It will generate 250 trajectories of 800 timesteps each and stores in `trajectoriesData.pickle` file
and use these data to train the agent automatically.

```
	python3 main.py train
```

It will load the `trajectoriesData.pickle` already in the folder and train the network

```
	python3 main.py showcells
```

It will use the trained agent to generate a trajectory of 1 milion timsteps and show the **activity maps**.

## Result


## Sources
* [Nature paper](https://www.nature.com/articles/s41586-018-0102-6) by Deepmind
* [Deepmind article](https://deepmind.com/blog/grid-cells/)
* [What are Grid cells? article about the paper]() (add link)
