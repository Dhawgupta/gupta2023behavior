# Code for Paper Behaviour Alignment Using Learned Reward Functions

The code is prganized into different folders for each experiment. The code for the experiments in the paper can be found in the following folders:


1. `barfi-cp` : Code for CartPole Experiment
2. `barfi-gw` : Code for GridWorld Experiment
3. `barfi-mc` : Code for MountainCar Experiment    
4. `barfi-mujoco` : Code for Mujoco Experiment


The configurations of a experiment is specified using `json` files, which can be found in the `experiments` folder of each experiment. To run a configuration(lets for example in GridWorld).

```bash
cd barfi-gw
python src/main.py experiments/PaperPlots/good/barfineumann.json 0
```
To run the 0th configuration in the `experiments/PaperPlots/good/barfineumann.json` file. The results are stored in the `results` folder. 

```bash
python run/pending.py -j experiments/PaperPlots/good/barfineumann.json 
```
To find all the pending experiments to be run for the `barfineumann.json` file. 

```bash
python run/local.py -p src/main.py -j experiments/PaperPlots/good/barfineumann.json -c 8
```
To run all configurations in the `barfineumann.json` file using 8 threads in parallel. 

After completely running a configuration file, first have to process the data in order to average over all the seeds

```bash
python analysis/process_data.py  experiments/PaperPlots/good/barfineumann.json
```

Then to plot the results

```bash
python analysis/learning_curve_plot_final.py  experiments/PaperPlots/good/barfineumann.json
```


Each folders `experiments` directory has a folder called `PaperPlots` which can be used to reproduce the plots in the paper. The `good` usually refers to good aux reward and `bad` refers to bad aux reward.

