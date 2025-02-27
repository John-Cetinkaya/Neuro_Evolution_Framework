# Neurological Evolution
This project is still a WIP but currently as is can run a and learn to play Cart Pole from the Gymnasium package. It uses inspiration from Pezzza's Work and the NEAT paper to train a reinforcement learning model.

**NEAT paper:** http://nn.cs.utexas.edu/downloads/papers/stanley.cec02.pdf
**Pezzza's Work:** https://www.youtube.com/@PezzzasWork

## Idea behind the project:
The plan for this project was to use Genetic Algorithms to train an AI to play some kind of game which naturally brought me to NEAT. While I could have just used the NEAT for python package I wouldn't have learned nearly as much about how the inner working really work. So I elected to make my own framework while cutting out the crossing over of neural networks in the interest of time.

## Dependencies
This was build on python 3.12.7 with:
gymnasium == 1.1.0
numpy == 2.1.2
networkx 3.4.2
matplotlib == 3.8.4

This project is by no means fast or efficient it was built with curiosity in mind and not too much stress. I do plan to continue on with adding multiprocessing and working with other environments to see what can really be figured out. I have a lot of analysis left to figure out issues with mutation and what is the more efficient as well as how to stop networks from growing uncontrollably. Speed is now a larger concern cause as I add features or try and test various aspects things will start to slow down.

**To test:** If all dependencies are installed you should be able to go to run_it.py and run it.