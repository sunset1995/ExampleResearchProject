# ExampleResearchProject

My resarch project initial point.

#### New dataset work flow
1. Create a new file `lib/dataset/[NEW_DATASET].py`.
2. Implement the dataset class in `lib/dataset/[NEW_DATASET].py`.
3. Update `lib/dataset/__init__.py`.

The implemented dataset should return a dictionary as the batch for training. 
All key in the batch is preserved for the network.

See `lib/dataset/dataset_example.py` for an example.

#### New model work flow:
1. Create a new file `lib/model/[NEW_MODEL].py`.
2. Implement the network in `lib/model/[NEW_MODEL].py`.

The implemented network should have a member function `def compute_losses(self, batch: dict) -> dict:`
where the batch is implemented by yourself in `lib/dataset/[NEW_DATASET].py`.
The `compute_losses` function should return a dictionary
where all element will be accumulate and log at each epoch.
The key `total' will be backprob in `train.py`.

See `lib/model/Example.py` for an example.

#### New experiment work flow
1. Copy `config/ExampleTask/example.yaml` and alter the field to match your implemented dataset/model and all other training detail.
2. `python train.py --cfg [PATH/TO/YOUR.yaml]`

Run `python train.py --cfg config/ExampleTask/example.yaml` for an example.
