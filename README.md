# Reinforcement Learning - Obstacle Tower Project  <!-- omit in toc -->
- [Team](#team)
- [Setup](#setup)
  - [Packages](#packages)
  - [Environment](#environment)
    - [Environment Configuration](#environment-configuration)
- [Evaluating](#evaluating)
- [Training](#training)
- [Approach](#approach)


In this project we had to create an agent to tackle the [Obstacle Tower Challenge](https://github.com/Unity-Technologies/obstacle-tower-env).  The agent must ascend a tower, proceeding through as many floors/levels as possible.

# Team
* Nishai Kooverjee      (135477)
* Kenan Karavoussanos   (1348582)
* Angus Mackenzie       (1106817)
* Africa Khoza          (1137682)

# Setup
To run this code, you need to have the requisite packages and the environment setup.

## Packages
To install the packages, run the following command:
```
conda env create -f environment.yml
```
Then activate the environment by running:
```
conda activate proj
```

## Environment
This project required an offshoot of the obstacle tower environment. The environment is too large for github, so we had to save it on google drive. Download the `ObstacleTower.zip` file from [Google Drive](https://drive.google.com/open?id=1LYwM_Qnn7mhRadTO8g9thmSbIxXmRGpu), and then unzip it into the repository's directory. You will likely need to change the permissions in order to make it executable, you can do this by running the following in the repository directory.
```
chmod -R 755 ./ObstacleTower/obstacletower.x86_64
```

### Environment Configuration

The following configuration was laid out for us in the course:
```
starting-floor':        0
total-floors':          9
dense-reward':          1
lighting-type':         0
visual-theme':          0
default-theme':         0
agent-perspective':     1
allowed-rooms':         0
allowed-modules':       0
allowed-floors':        0
```

# Evaluating
To get an estimate of the score obtained by the agent during the marking, you can do the following.


Before attempting an evaluation, ensure the `MyAgent.py` file's `__init__` method has the path to load the weights from, an example follows:
```python
self.policy_network.load_state_dict(torch.load("checkpoints/40000.pth",map_location=torch.device(device)))
```
Where `"checkpoints/40000.pth"` is the location of our model's weights.

Then to run the evaluation script:
```
python evaluation.py --realtime
```
This will run the `evaluation.py` script on 5 different seeds, and will return the score gained across those runs. The `--realtime` flag indicates whether the environment will be rendered so you can watch the trial happening. If you do not want to watch the trial, and want to get the results as fast as possible, simply run the command without the `--realtime` flag.

# Training
To train a new agent simply run:
```
python train_atari.py --checkpoint checkpoints/40000.pth
```
You can remove the `--checkpoint` flag if you want to train one from scratch and not use any pretrained weights.

The above command will create a new folder, called `results/experiment_1`, and will store the rewards attained as well as checkpoints in that folder. For each new run of `train_atari.py` a new `experiment_<n>` folder will be created.


# Approach
We used a Deep Q Network as the backbone of our agent. The code was largely based off one of our [previous assignments](https://github.com/AngusTheMack/dqn-pong). We used minimal wrappers, and simply trained a number of models over the course of few weeks. Often using a pretrained model's weights to initialise another model, and changing different hyperparameters along the way. We reached level 5 in the tower, and achieved a score of 40000. Considering the aim was to beat an agent with a score of 8000, we did notable well. This assignment has a [leader board](https://moodle.ms.wits.ac.za/piedranker/app/php/rankings.php?assignid=431&courseid=74) so that students can track how their agents compare against others, and some students achieved truly remarkable performance.
