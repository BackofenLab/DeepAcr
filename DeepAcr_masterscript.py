import torch
from torch_geometric.data import DataLoader
import learn_test_functions
import sys, getopt
import glob
import random
import numpy as np
import uuid
from argparse import ArgumentParser
import database_mem
import pathlib
import pandas as pd
import sys
sys.path.insert(0, './')

import sys


import psutil
import LSTM as LSTM
import LSTMb as LSTMb
import GRU as GRU

import GRUb as GRUb

import Linear as Linear
import Linearb as Linearb
             
             
def isBetween(value, intervall):


    return intervall[1] < value <= intervall[0]



class MyDataset():

    def __init__(self, root: str, input_files: list, max_norm: list = [None], max_length: list = [None], shuffle: bool = True, start_label: int = 0):

        self.shuffle = shuffle
        self.max_norm = max_norm
        self.file_list = input_files
        self.path_way = root
        self.num_features = 21
        self.prot_start_label = start_label
        self.mean = []
        self.std = []
        self.max_length = 0
        self.max_length_input = max_length
        self.filenames = [self.path_way + file for file in input_files]
        self.class_length = len(input_files)




    def process(self):
        # Read data into huge `Data` list.

            data_list = []
            protein_label = self.prot_start_label
            z_list = []
            max_length_list = []
            data_strings = []

            for file_name in self.filenames:
                data_, max_length = self.load_protein_data(file_name)
                data_strings.append(data_)
                max_length_list.append(max_length)

            max_length_list = [self.max_length_input] if np.array(self.max_length_input).all() != None else max_length_list

            for prot_data in data_strings:
                for c in range(len(prot_data[0])):
                
                    try:
                                        
                        graph_data, z_data = self.create_graph(protein_label,[prot_data[0][c],prot_data[1][c]], max_ =  max_length_list)
                        z_list.append(z_data)
                        data_list.append(graph_data)


                    except:
                        continue
                        
                protein_label = protein_label + 1

            max_data = np.max(z_list, axis=0) if np.array(self.max_norm).all() == None else self.max_norm
            data_list = self.adjust_zdata(data_list, max_data)

            if self.shuffle:
                random.shuffle(data_list)

            self.max_length =np.max(max_length_list)


            return data_list,max_data




    def load_protein_data(self, file: str):

        protein_list_full = []
        protein_id_full = []
        protein_list = []
        protein_id = []

  
        protein_list_full.extend(protein_list)
        protein_id_full.extend(protein_id)
        max_len = np.max([len(x) for x in protein_list_full])

        return [protein_list_full, protein_id_full], max_len




    def adjust_zdata(self, data_list:list, max_data:list):

        data_list = [[data[0], data[1],data[2]/max_data, data[3],data[4]] for data in data_list]
        data_list = [Data(x=data[0], y=data[1], z= data[2].float(), edge_index=data[3].t().contiguous(), id = data[4]) for data in data_list]


        return data_list



    def read_yaml(self):

        try:

            with open(r'.') as file:
                documents = yaml.full_load(file)
        except FileNotFoundError:
            print("File not accessible")
            sys.exit(0)

        self.num_features = len(documents)

        return documents




    def create_graph(self, prot_type,protein, max_ = None):
        dic_amino = self.read_yaml()
        max_length = np.max(max_)

        edge_list = []
        onehot_encoded = []
        prot_id, string_graph = protein[1], protein[0]
        l_length = len(string_graph)
        
        print(string_graph)
        

        onehot_encoded = onehot_encoded[:max_length]
        to_pad = max_length - len(onehot_encoded)

        for i in range(to_pad):
            listofzeros = [0] * size
            onehot_encoded.append(listofzeros)

        edge_index = torch.tensor(edge_list, dtype=torch.long)
        x = torch.tensor(onehot_encoded, dtype=torch.float)
        y = torch.tensor([prot_type], dtype=torch.int)
        z = torch.tensor(z_data,dtype = torch.float)

        data = [x,y,z,edge_index,str(prot_id)]

        return data, z_data


if __name__ == '__main__':


    parser = ArgumentParser()

    parser.add_argument("-f", "--file", dest="filename",
                    help="protein files to be read in", metavar="FILE")
    parser.add_argument("-b", "--batch_size", dest="batch_size",
                    help="Batch size for training", type=int, default = 30)
    parser.add_argument("-d", "--data_path", dest="data_path",
                    help="path of the data", default = "files")
    parser.add_argument("-m", "--model_path", dest="model_path",
                    help="path of the models", default = "models")
    parser.add_argument("-p", "--prediction_path", dest="prediction_path",
                    help="path of the predictions", default = "prediction")
    parser.add_argument("-g", "--helper", dest="helper",
                    help="helper", type = int)
                    
                    
        
               

    args = parser.parse_args()
    
        
    functions = {
             'LSTMa': LSTM.LSTM,
             'LSTMb': LSTMb.LSTMb,
              'Lineara': Linear.Linear,
              'Linearb': Linearb.Linearb,
             'GRUa': GRU.GRU,
             'GRUb': GRUb.GRUb}

    if args.filename is None:
        print("Provide a file to run the prediction using -f 'filename'")
        sys.exit(0)

    pathlib.Path(args.prediction_path).mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.prediction_path + 'with_infos').mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.prediction_path + 'without_infos').mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.prediction_path +'with_infos/LSTM').mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.prediction_path + 'with_infos/LINEAR').mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.prediction_path + 'with_infos/GRU').mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.prediction_path + 'without_infos/LSTM').mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.prediction_path + 'without_infos/LINEAR').mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.prediction_path + 'without_infos/GRU').mkdir(parents=True, exist_ok=True)

    BATCH_SIZE = args.batch_size
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_list = []
    model_names = []

    shuffle = False
    print("Prediction: " + str(args.filename))
    file_list = args.filename.split(",")
    files = [args.data_path + f for f in file_list]
    print(files)

    model_names_list = [["LSTM","LSTM", "LSTM"], ["GRU", "GRU", "GRU"], ["LINEAR", "LINEAR", "LINEAR"]]
    with_infos = [True, False]

    path = args.model_path
    models_with_infos = torch.load(path + "/models_with_infos.pt")
    models_without_infos = torch.load(path + "/models_without_infos.pt")
    models_dict = {True: models_with_infos, False: models_without_infos}
    
    
    

    for infos in with_infos:
        write_ids = True
        frames = []

        for model_names in model_names_list:
            print(model_names)
            run_first = True

            checkpoint_list = []

            for model_num, model_name in enumerate(model_names):
                checkpoint = models_dict[infos][model_name + "_" + str(model_num)]
                max = checkpoint["data_max"]
                max_length = checkpoint["data_max_length"]
                checkpoint_list.append(checkpoint)
                
                
            database = database_mem.MyOwnDataset(root="", input_files  = files, max_norm= max, max_length = max_length, shuffle = shuffle)
            dataset, data_max = database.process()
            
            print("dataset")
            print(dataset)
            

            model_list = []

            for checkpoint in checkpoint_list:

                type_name = str(checkpoint["model_type"]).split(" ")[1].split(".")[1]
                type_name = str(type_name) + "a" if type_name[-1] != "b" else type_name
                model = functions[type_name](database.num_features, max_length).to(device)
                model.load_state_dict(checkpoint["model"])
                model_list.append(model)



            test_loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=False)
            df_list = []
            for batch in test_loader:
                df1 = pd.DataFrame()
                dataframe =learn_test_functions.evaluate_proteins(batch, model_list, run_first, model_names, with_infos = infos, write_ids = write_ids, prediction_path = args.prediction_path)
                run_first =  False
                df_list.append(dataframe)
                


            df1 =  pd.concat(df_list, axis=0)
            write_ids = False
            frames.append(df1)
            

        result = pd.concat(frames, axis=1)

        """
        add majority vote
        """

        ensemble_predictions = []
        #for "ensamble*" in result.keys():
        keys = list(filter(lambda x: x.startswith("ensamble"), result))
        ensemble_predictions =  [result[k].tolist() for k in keys]
        keys = list(filter(lambda x: x.endswith("negative_prediction"), result))
        ensemble_pred1 =  [result[k].tolist() for k in keys]
        keys = list(filter(lambda x: x.endswith("positive_prediction"), result))
        ensemble_pred0 =  [result[k].tolist() for k in keys]

        majority_vote = [round(np.mean([ensemble_predictions[l][i] for l in range(len(ensemble_predictions)) ])) for i in range(len(ensemble_predictions[0]))]
        average_class1 = [np.mean([ensemble_pred1[l][i] for l in range(len(ensemble_pred1)) ]) for i in range(len(ensemble_pred1[0]))]
        average_class0 = [np.mean([ensemble_pred0[l][i] for l in range(len(ensemble_pred0)) ]) for i in range(len(ensemble_pred0[0]))]
        average_pred = [np.argmax([average_class1, average_class0], axis = 0)]



        result["majority vote"] = majority_vote
        result["average_positive_prediction"] = average_class0
        result["average_negative_prediction"] = average_class1
        result["average_prediction"] = list(average_pred)[0]
        
        conf_dict = {"high":[1.01, 0.80] , "medium": [0.80, 0.65], "low": [0.65, 0.50]}
        
        confidence = []
        for max_value in np.max([average_class1, average_class0], axis = 0):
            confidence.extend([k for k in list(conf_dict.keys()) if isBetween(max_value, conf_dict[k])])
            
        result["confidence"] = confidence
        
        

        append_string = "with_infos" if infos else "without_infos"
        result.to_csv(path_or_buf=args.prediction_path + append_string + "/" + "summary"  + ".csv", mode = "a", header = True)
