import glob
import os
import torch
from torch_geometric.data import DataLoader
import os
import glob





class Saver():

    def __init__(self, model_path = "./models"):
        self.model_path = model_path + "/"




    def save_in_dict(self,saved_model, optimizer, epoch, data_max, data_max_length, number, net_name):

        save_model_path = self.model_path
        model = save_model_path + net_name + str(number) +  ".pt"

        torch.save({'model_type':type(saved_model),
                    'epoch' : epoch,
                    'model': saved_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'data_max': data_max,
                    'data_max_length':data_max_length},
                    model)

        return





    def save_models_together(self):

        model_list = []
        model_names = []
        model_names_list = [["LSTM", "LSTM", "LSTM"], ["GRU", "GRU", "GRU"], ["LINEAR", "LINEAR", "LINEAR"]]
        with_infos = [True, False]

        model_list1 = {"LSTM_1": [], "LSTM_2": [], "LSTM_3": [], "GRU_1": [], "GRU_2": [], "GRU_3": [], "LINEAR_1": [],
                       "LINEAR_2": [], "LINEAR_3": []}
        model_list2 = {"LSTM_1": [], "LSTM_2": [], "LSTM_3": [], "GRU_1": [], "GRU_2": [], "GRU_3": [], "LINEAR_1": [],
                       "LINEAR_2": [], "LINEAR_3": []}

        for infos in with_infos:
            write_ids = True
            frames = []

            for model_names in model_names_list:
                run_first = True

                path = self.model_path + model_names[0]
                models = glob.glob(path + "b*" if infos else path + "a*")
                model_list.extend(models)
                checkpoint_list = []

                for model_num, model_name in enumerate(models):
                    checkpoint = torch.load(model_name)
                    model_type = checkpoint["model_type"]
                    model = checkpoint["model"]
                    optimizer = checkpoint["optimizer"]
                    epoch = checkpoint["epoch"]
                    max = checkpoint["data_max"]
                    max_length = checkpoint["data_max_length"]
                    checkpoint_list.append(checkpoint)

                    if infos == True:
                        key = model_names[0] + "_" + str(model_num)

                        model_list1[key] = checkpoint
                    else:
                        key = model_names[0] + "_" + str(model_num)
                        model_list2[key] = checkpoint

        model_path = self.model_path
        torch.save(model_list1, model_path + "models_with_infos.pt")
        torch.save(model_list2, model_path + "models_without_infos.pt")


        for f in model_list:
            os.remove(f)

        return


