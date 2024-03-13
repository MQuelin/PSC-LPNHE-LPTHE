import torch

from pathlib import Path

from flows import ConditionalNF
from datasets import ZeeDataset
from train import ConditionalTrainer


import matplotlib.pyplot as plt

layer_counts = [16,32,64]
MLP_shape_lists = [[80,80],[160,160],[320,320],[160,160,160],[80,80,80,80],[60,60,60,60,60,60]]

test_folder = "/test_3" #Modify the name of this folder to separate different tests, don't forget the first ' / '

absolute_path = Path(__file__).parent / ("../models/parameter_testing" + test_folder + "/test_results.txt")

data = ZeeDataset('../data/100k2.pkl')
device = "cuda" if torch.cuda.is_available() else "cpu"
learning_rate = 5e-4
epsilon = 0.05
nb_epochs = 30
batch_size = 1000
percentage_train = 0.8

final_testing_errors = []

with open(absolute_path, 'w') as f:
    f.write('Testing for parameter influence on model efficiency\n\n\n')
    f.write('training on 100k2.pkl\n')
    f.write(f'espilon = {epsilon}, lr = {learning_rate}, batch_size= {batch_size}, train/test ratio : {percentage_train}\n\n')
    f.write('parameters explorered :\n\n')
    f.write('layer_counts : ' + str(layer_counts) + '\n')
    f.write('MLP_shape_lists : ' + str(MLP_shape_lists) + '\n\n\n')
    f.write('____________________________________________________\n')

    index = 0 #index is used to easily reference a model
    for layer_count in layer_counts:
        for MLP_shape_list in MLP_shape_lists:
            flow = ConditionalNF(layer_count, 3, 10, MLP_shape_list)
            optimizer = torch.optim.Adam(flow.parameters(), lr=learning_rate)

            ### Dividing the data on train and test
            train_len = int(percentage_train * len(data))
            data_train, data_test = torch.utils.data.random_split(data, 
                                                                [train_len, len(data)-train_len])

            

            dataloader = torch.utils.data.DataLoader(data_train, batch_size=batch_size, shuffle=True)
            dataloader_test = torch.utils.data.DataLoader(data_test, batch_size=batch_size, shuffle=True)

            trainer = ConditionalTrainer(flow, optimizer, dataloader, dataloader_test, 3, 10, epsilon=epsilon, device=device)
            
            loss_train, loss_test = trainer.train(nb_epochs)

            model_name = f"CNF_{layer_count}lyrs_{len(MLP_shape_list)}x{MLP_shape_list[0]}shape_100k2Zee_{epsilon}_060324.pt"

            trainer.save_at(save_path= "../models/parameter_testing" + test_folder, save_name=model_name)

            final_testing_errors.append(loss_test[-1])

            f.write(model_name + '\n')
            f.write(f'trained for {nb_epochs} epochs' + '\n\n')
            f.write(f'model index : i={index}\n\n')
            f.write(f"final loss = {loss_test[-1]}\n\n")
            f.write('training loss exerpt (in steps of 20 to reduce amount of data displayed) :\n')
            f.write( str(loss_train[::20]))
            f.write('\n\ntesting loss exerpt (in steps of 5 to reduce amount of data displayed) :\n')
            f.write( str(loss_test[::5]))
            f.write('\n\n____________________________________________________\n')

            index = index + 1

    f.write('\n successfuly exited full training and testing loop\n\n')
    f.write('final testing errors, ordered by model index :\n')
    f.write(str(final_testing_errors) + '\n\n')
    f.write(f'minimal error reached for model of index i={final_testing_errors.index(max(final_testing_errors))}')
    f.close()