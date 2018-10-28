# Vector-based Navigation using Grid-like Representations in Artificial Agents

## About
Replicating Google Deepmind's paper ["Vector-based Navigation using Grid-like Representations in Artificial Agents"](https://deepmind.com/blog/grid-cells/).

[What are Grid cells?  Full article about the paper here]() (add link)

## Dependencies
* Tensorflow
* Numpy
* Matplotlib

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
