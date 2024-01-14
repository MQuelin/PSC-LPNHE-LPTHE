## File structure:

data contains ressources used as training data
models is the default file where models trained using code in the src file are saved
src contains the source python code:
- bijective_transforms.py contains functions that can be used as layers in a Normalizing flow
- data_loaders.py contains code used to load files in the data directory to be used as training sets
- flow_utils.py contains miscellaneous ressources used by other parts of the code
- flows.py contains model architectures
- misc_transforms.py contains transforms that aren't bijective and can be used by other part of the code (like MLPs for example)
- train.py contains code used to perform training and save the models

- example_trainer.py is and example of how to train a simple model.

- tests.py is used for any testing purposes during coding -> needs to be added to the .gitignore



## Useful ressources and videos:

Normalizing flow code example
https://www.youtube.com/watch?v=OiwtJA9In1U
https://github.com/MaximeVandegar/Papers-in-100-Lines-of-Code/tree/main/Variational_Inference_with_Normalizing_Flows

Examples of normalizing flows
https://deepgenerativemodels.github.io/notes/flow/

Mathematical theory
https://www.youtube.com/watch?v=i7LjDvsLWCg
