import torch

from pathlib import Path

from flows import ConditionalNF
from datasets import ZeeDataset
from train import ConditionalTrainer


import matplotlib.pyplot as plt

layer_counts = [4,8,16,32,64,128,256,512]
MLP_shape_lists = [[10,10],[20,20],[40,40],[80,80],[10,10,10],[80,80,80],[20,20,20,20]]

test_folder = "/test_1" #Modify the name of this folder to separate different tests, don't forget the first ' / '

absolute_path = Path(__file__).parent / ("../models/parameter_testing" + test_folder + "/test_results.txt")

data = ZeeDataset('../data/100k2.pkl')
device = "cuda" if torch.cuda.is_available() else "cpu"
learning_rate = 5e-4
epsilon = 0.05
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

    k = 0 #k is used to count the current index of layer_count in which we are currently to only try all MLP_shapes sometimes and shorten runtime
    index = 0 #index is used to easily reference a model
    for layer_count in layer_counts:
        if k%3 == 0:
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
                
                # during testing we observed that the required number of epochs needed for training decrease over time, hence a variable epoch number
                loss_train, loss_test = trainer.train(int(20.-10*(layer_count-24)/(512.-24.)))

                model_name = f"CNF_{layer_count}lyrs_"
                for MLP_shape in MLP_shape_list:
                    model_name = model_name + str(MLP_shape) + "_"
                model_name = model_name + f"shape_100k2Zee_{epsilon}_060324.pt"

                trainer.save_at(save_path= "../models/parameter_testing" + test_folder, save_name=model_name)

                final_testing_errors.append(loss_test[-1])

                f.write(model_name + '\n')
                f.write(f'trained for {int(20.-10*(layer_count-24)/(512.-24.))} epochs' + '\n\n')
                f.write(f'index : {index}\n\n')
                f.write(f"final loss = {loss_test[-1]}\n\n")
                f.write('training loss exerpt (in steps of 10 to reduce amount of data displayed) :\n')
                f.write( str(loss_train[::10]))
                f.write('\n\ntesting loss exerpt (in steps of 3 to reduce amount of data displayed) :\n')
                f.write( str(loss_test[::3]))
                f.write('\n\n____________________________________________________\n')

                index = index + 1

        else :
            flow = ConditionalNF(layer_count, 3, 10, [20,20])
            optimizer = torch.optim.Adam(flow.parameters(), lr=learning_rate)

            ### Dividing the data on train and test
            train_len = int(percentage_train * len(data))
            data_train, data_test = torch.utils.data.random_split(data, 
                                                                [train_len, len(data)-train_len])

            

            dataloader = torch.utils.data.DataLoader(data_train, batch_size=batch_size, shuffle=True)
            dataloader_test = torch.utils.data.DataLoader(data_test, batch_size=batch_size, shuffle=True)

            trainer = ConditionalTrainer(flow, optimizer, dataloader, dataloader_test, 3, 10, epsilon=epsilon, device=device)

            # during testing we observed that the required number of epochs needed for training decrease over time, hence a variable epoch number
            loss_train, loss_test = trainer.train(int(20.-10*(layer_count-24)/(512.-24.)))

            model_name = f"CNF_{layer_count}lyrs_20_20_shape_100k2Zee_{epsilon}_060324.pt"

            trainer.save_at(save_path= "../models/parameter_testing" + test_folder, save_name=model_name)

            final_testing_errors.append(loss_test[-1])

            f.write(model_name + '\n')
            f.write(f'trained for {int(20.-10*(layer_count-24)/(512.-24.))} epochs' + '\n\n')
            f.write(f"final loss = {loss_test[-1]}\n\n")
            f.write('training loss exerpt (in steps of 10 to reduce amount of data displayed) :\n')
            f.write( str(loss_train[::10]))
            f.write('\n\ntesting loss exerpt (in steps of 3 to reduce amount of data displayed) :\n')
            f.write( str(loss_test[::3]))
            f.write('\n\n____________________________________________________\n')

            index = index + 1

        k = k+1

    f.write('\n successfuly exited full training and testing loop\n\n')
    f.write('final testing errors, ordered by model index :\n')
    f.write(str(final_testing_errors))
    f.close()