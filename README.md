# The official implementation of paper "Neural relational and dynamics inference for complex systems"



### Requirements
* Pytorch 0.2 (0.3 breaks simulation decoder)
* Python 2.7 or 3.6

### Data generation

To replicate the experiments on simulated physical data, first generate training, validation and test data by running:

```
cd data
python generate_dataset.py
```
This generates the springs dataset, use `--simulation charged` for charged particles.


### Run experiments

From the project's root folder, simply run
```
python train.py
```
to train a NRDI model on the springs dataset. You can specify a different dataset by modifying the `suffix` argument: `--suffix _charged5` will run the model on the charged particle simulation with 5 particles (if it has been generated).
You can specify a different undersampling ratio by modifying the '--sample-percentage' argument: '--sample-percentage 0.8' will undersample 0.8 percentage of time steps from the trajectories.

Additionally, we provide code for an LSTM baseline (denoted *LSTM (joint)* in the paper), which you can run as follows:
```
python lstm_baseline.py
```

Finally, we also provide the code for experiment 2 (network reconstruction), which simply requires running the following code:
```
python exp2_train.py
```
