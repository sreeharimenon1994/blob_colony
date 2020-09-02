# Multi-Task Reinforcement Learning for Multi-Agent Colony 

Multi-task deep reinforcement learning is being used to analyze the behaviour of the multiple agents according to various stimulus from the environment. From a practical perspective, endowing the deep network an ability to interactively solve problems by communication would make them more adaptable and useful in our daily life. The paper intends to study communication, adaptability and how the agents evolve and survive depending on the different constraints that occur in the environment.

* To train `python main.py`
* To visualise `python visualise.py` (after training change the model name in visualise.py)


![working](/visualise_itr/working.png)

## Installation

pip:

    pip install -r requirements.txt

PyTorch:


    Windows: 
    pip install torch===1.6.0 torchvision===0.7.0 -f https://download.pytorch.org/whl/torch_stable.html

	Linux: 
	pip install torch torchvision


## Model Architecture

![arch](/visualise_itr/arch.png)
