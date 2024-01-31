"""
This code is to put at the beginning of your code to extract pickle dictionnaries
pickle needs to be installed on the computer
"""

import pickle

path_to_dict = "I:/PSC_CODE/Data/dictionnaire_300.pkl" # Replace this by the string to your pickle, example : "I:/PSC_CODE/Data/dictionnaire_300.pickle"

with open(path_to_dict, 'rb') as f:
    dico = pickle.load(f)

#%% Rest of your code below