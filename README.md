此分支基于spectre仿真
## Notice
This code requires Anaconda / miniconda. 

After installing Conda, you need to set up the environment.  
By default, `conda init` sets up Conda for `bash`, but you can configure it for `csh` by adding the following lines to your `csh` configuration file:
```bash
#add these to the csh file, which means your 'cshrc.gf55BCDlite_v1090' file
# Conda setup
setenv PATH "${HOME}/anaconda3/bin:${PATH}" # set all these path to you conda installation path
if ( -f "${HOME}/anaconda3/etc/profile.d/conda.csh" ) then
    source "${HOME}/anaconda3/etc/profile.d/conda.csh"
else
    echo "No Conda csh file found in the specified path."
endif
```
To start conda environment, first run:

```bash
conda activate
```
which will start the conda base, then follow the setup instruction.

Additionally, the default netlist file is located at:
`eval_engines/ngspice/ngspice_inputs/netlist/2opamp`

## Setup

In order to obtain the required packages, run the command below from the top level directory of the repo to install the `Anaconda` environment:

```bash
conda env create -f environment.yml
```

To activate the environment run:
```bash
source activate autockt
```
You might need to install some packages further using pip if necessary. To ensure the right versions, look at the `environment.yml` file.

## Code Setup
The code is setup as follows:

<img src=readme_images/flowchart.png width="500">

The top level directory contains two sub-directories:
* AutoCkt/: contains all of the reinforcement code
    * val_autobag_ray.py: top level RL script, used to set hyperparameters and run training
    * rollout.py: used for validation of the trained agent, see file for how to run
    * envs/ directory: contains all OpenAI Gym environments. These function as the agent in the RL loop and contain information about parameter space, valid action steps and reward.
* eval\_engines/: Contains all of the code pertaining to simulators
    * ngspice/: this directory runs all NGSpice related scripts.
        * netlist_templates: the exported netlist file modified using Jinja to update any MOS parameters
        * specs_test/: a directory containing a unique yaml file for each circuit with information about design specifications, parameter ranges, and how to normalize. 
        * script_test/: directory with files that test functionality of interface scripts  

## Training AutoCkt
Make sure that you are in the Anaconda environment. Before running training, the circuit netlist must be modified in order to point to the right library files in your directory. To do this, run the following command:

To generate the design specifications that the agent trains on, run:
```bash
python autockt/gen_specs.py --num_specs ##
```
The result is a pickle file dumped to the gen_specs/ folder.

To train the agent, open `ipython` from the top level directory and then: 
```bash
run autockt/val_autobag_ray.py
```
The training checkpoints will be saved in your home directory under ray\_results. Tensorboard can be used to load reward and loss plots using the command:

```bash
tensorboard --logdir path/to/checkpoint
```

To replicate the results from the paper, num_specs 350 was used (only 50 were selected for each CPU worker). Ray parallelizes according to number of CPUs available, that affects training time. 
## Validating AutoCkt
The rollout script takes the trained agent and gives it new specs that the agent has never seen before. To generate new design specs, run the `gen_specs.py` file again with your desired number of specs to validate on. To run validation, open `ipython`:

```bash
run autockt/rollout.py /path/to/ray/checkpoint --run PPO --env opamp-v0 --num_val_specs ### --traj_len ## --no-render
``` 
* num_val_specs: the number of untrained objectives to test on
* traj_len: the length of each trajectory

Two pickle files will be updated: `opamp_obs_reached_test` and `opamp_obs_nreached_test`. These will contain all the reached and unreached specs, respectively.

## Results
Please note that results vary greatly based on random seed and spec generation (both for testing and validation). An example spec file is provided that was used to generate the results below. 

<img src=readme_images/results.png width="800">

The rollout generalization results will show up as pickle files `opamp_obs_reached_test` and `opamp_obs_nreached_test`. For this particular run, we obtained 938/1000. Additional runs were also conducted, and we found that the results varied from 80%-96% depending on the generated specs during rollout, and the specs that were changed during training. Our results were obtained by running on an 8 core machine, we've found that running on anything below 2 cores results in weird training behavior. 

## Cascode modify
Modifying requires editors to edit paramters at line 121 in ngspice_vanilla_opamp.py(self.cur_params_idx = np.array([33, 33, 33, 33, 33, 14, 20])),editors are required to modify the initialized paramters due to your own requirements.
Secondly, editors need to edit env.step([2, 2, 2, 2, 2, 2, 2]) because you need to set the initialized steps for the agent. Basically, the numbers should be set 2 as positive steps. The number of paramters is set due to your own requirements.
Thirdly, editors have to edit yaml file. In this project, two_stage_opamp.yaml should be modified. Editors have to set your own paramters and modify the target_spec as you expect.
