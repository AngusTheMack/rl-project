# DQN Variations Project

# Setup
Run the following command to create a new conda env with the required pacakges:
```
conda env create -f environment.yml
```
If you install more packages and need to add them to the `.yml` file, you can use the command:
```
conda env export > environment.yml
```

# Team
* Nishai Kooverjee
* Kenan Karavoussanos
* Angus Mackenzie
* Africa Khoza



# Assignment Outline
* Implement Three Deep RL algorithms on Three Atari Games
* Submit code and write-up that analyses these three algorithms on the games we picked
* Need to provide a detailed write-up of the results, which will serve as the basis for the assignment mark
* Analyse the algorithms by performing a sensitivity analysis on a single hyperparameter of our choosing. The hyperparameter can be a standard parameter, like learning rate, or the network architecture, activation function, etc. Run the experiments with different reasonable values and determine whether the results change significantly. 
  * Comment on our findings.
* The write-up should consists of:
  * Detailed description of the algoriths, and how they differ from one another
  * Training results for each of the algorithms should be graphed and shown, one for each game
  * Multiple runs of each algorithm should be performed and the results averaged to avoid high variance and to make the results statistically significant
  * Provide a discussion based on these results, identifying strengths and weakness and general trends observed 

# To Do
- [ ] Read The Papers
- [ ] Implement DQN on 3 Environments
- [ ] Implement DDQN on 3 Environments
- [ ] Implement Duelling DQN on 3 Environments
- [ ] Decide on hyperparameter to alter
- [ ] Run models to get results
- [ ] Write-up results