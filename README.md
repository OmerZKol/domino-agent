# domino-agent
## Overview
- Contains a custom 3&5 Domino environment to train a RL agent
- Training environment made using stable baseline3 and gymnasium 
- Contains some already trained models and their training logs

## Contents
- DominoGame.py contains the Domino Game, can be run to just have 2 players play against eachother (default is a random player)
- DominoEnv.py Contains the RL training Environment
- TrainAgent.py trains and saves an agent
- TestAgent.py tests a trained agent against the random player and print the overall final results
- ContinueTrainingAgent.py takes a trained agent and continues training it

## Usage
- adjust the target_score in DominoEnv to change the target score of the domino game
- run TrainAgent.py to train an agent on the DominoEnv, log and model is saved in log and model file respectively
- run TestAgent.py after changing model directory to Agent to be tested to make it play against the randomplayer and output final overall score
- by default traines model for 500k steps (unless manually stopped)
- run ContinueTrainingAgent.py after changing the model directory and relevant log/model save directory to the agent to be trained
- by default traines model for 500k more steps (unless manually stopped)

## Installation
install dependencies using in command line:
```bash
>>> pip install -r requirements.txt
``` 
