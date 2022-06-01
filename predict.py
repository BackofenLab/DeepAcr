import torch
from torch_geometric.data import DataLoader
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
sys.path.insert(0, './Networks')
import argparse
import sys
import psutil
import Networks.LSTM as LSTM
import Networks.LSTMb as LSTMb
import Networks.GRU as GRU
import Networks.GRUb as GRUb
import Networks.Linear as Linear
import Networks.Linearb as Linearb




def evaluate_proteins(data, model_list, run_first, model_names, with_infos: bool = True, write_ids: bool = True, prediction_path: str = "./prediction/"):

    append_string = "with_infos" if with_infos else "without_infos"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    frames_df = pd.DataFrame()
    predictions = []

    model_pred = [[] for i in range(len(model_list))]
    model_val =  [[] for i in range(len(model_list))]
    dataframe = pd.DataFrame()

    for model_num, model in enumerate(model_list):
        model.to(device)
        model.eval()
        data.to(device)
        data_list = data.to_data_list()
        values = model_list[model_num](data)
        values = torch.sigmoid(values)
        pred = [1 if o > 0.5 else 0 for o in values]
        model_pred[model_num].extend(pred)
        predictions.append(values.tolist())
        model_val[model_num].extend(values.tolist())

    pred_v = np.mean(predictions, axis=0)
    pred = [1 if o > 0.5 else 0 for o in pred_v]
    append_string = "_with_infos" if with_infos else ""
    if write_ids:
        dataframe['protein_id' + append_string] = list(data.id)
        
        


    dataframe['ensamble_' + model_names[0]  + append_string] =  pred
    dataframe[model_names[0] + append_string+ "_positive_prediction" ] = [o[0] for o in pred_v]
    dataframe[model_names[0] + append_string+"_negative_prediction" ] = [1 - o[0] for o in pred_v]


    
    
    return dataframe

            
def isBetween(value, intervall):


    return intervall[1] < value <= intervall[0]

"""

  function main

  Function   : main logic of script
  Description: iterates over available ensembles to predict the data

"""



if __name__ == '__main__':


    parser = ArgumentParser()

    parser.add_argument("-f", "--file", dest="filename",
                    help="protein files to be read in", metavar="FILE")
    parser.add_argument("-b", "--batch_size", dest="batch_size",
                    help="Batch size for training", type=int, default = 30)
    parser.add_argument("-d", "--data_path", dest="data_path",
                    help="path of the data", default = "./")
    parser.add_argument("-m", "--model_path", dest="model_path",
                    help="path of the models", default = "./models")
    parser.add_argument("-p", "--prediction_path", dest="prediction_path",
                    help="path of the predictions", default = "./prediction/")
    parser.add_argument("-s", "--sequence_completenes", dest="sequence_completenes",
                    help="parial/x", default = "")
    parser.add_argument('--DNA', action=argparse.BooleanOptionalAction)
               

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


    BATCH_SIZE = args.batch_size
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_list = []
    model_names = []

    shuffle = False
    print("Predictiong: " + str(args.filename))
    file_list = args.filename.split(",")
    files = [args.data_path + f for f in file_list]


    model_names_list = [["LSTM","LSTM", "LSTM"], ["GRU", "GRU", "GRU"], ["LINEAR", "LINEAR", "LINEAR"]]
    with_infos = [True, False]

    path = args.model_path
    print(path)
    models_with_infos = torch.load(path + "/models_with_infos.pt")
    models_without_infos = torch.load(path + "/models_without_infos.pt")
    models_dict = {True: models_with_infos, False: models_without_infos}
    



    result_list = []
    
    

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
                
                
            
            database = database_mem.MyOwnDataset(root="", input_files  = files, max_norm= max, max_length = max_length, shuffle = shuffle, seq_complete = args.sequence_completenes,DNA = args.DNA)
            dataset, data_max = database.process()
            
            

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
                dataframe =evaluate_proteins(batch, model_list, run_first, model_names, with_infos = infos, write_ids = write_ids, prediction_path = args.prediction_path)
                run_first =  False
                df_list.append(dataframe)
                
                

            df1 =  pd.concat(df_list, axis=0)
            write_ids = False
            frames.append(df1)
            

        result = pd.concat(frames, axis=1)

        result_list.append(result)


        """
        add majority vote
        """
        
    result = pd.concat(result_list, axis=1)
    


    ensemble_predictions = []
    #for "ensamble*" in result.keys():
    keys = list(filter(lambda x: x.startswith("ensamble"), result))
    
    
    ensemble_predictions =  [result[k].tolist() for k in keys]
    keys = list(filter(lambda x: x.endswith("negative_prediction"), result))

    ensemble_pred1 =  [result[k].tolist() for k in keys]
    keys = list(filter(lambda x: x.endswith("positive_prediction"), result))

    ensemble_pred0 =  [result[k].tolist() for k in keys]

    resut_df_positive= pd.DataFrame()
    resut_df_negative = pd.DataFrame()
        
                
    only_positive_predictions = [np.max([ensemble_predictions[l][i] for l in range(len(ensemble_predictions)) ]) for i in range(len(ensemble_predictions[0]))]
        


    only_best_0_prediction = [np.max([ensemble_pred1[l][i] for l in range(len(ensemble_pred1)) ]) for i in range(len(ensemble_pred1[0]))]
    only_best_1_prediction = [np.max([ensemble_pred0[l][i] for l in range(len(ensemble_pred0)) ]) for i in range(len(ensemble_pred0[0]))]
        
        
    best_prediction = [only_best_1_prediction[enum] if x == 1 else only_best_0_prediction[enum] for enum, x in enumerate(only_positive_predictions)]
        

        
    conf_dict = {"high":[1.01, 0.80] , "medium": [0.80, 0.65], "low": [0.65, 0.50]}
        
        
    confidence = []
    for enum, best in enumerate(best_prediction):
        if only_positive_predictions[enum] == 1:
            confidence.extend([k for k in list(conf_dict.keys()) if isBetween(best, conf_dict[k])])
        else:

            confidence.append("low")

                

    
    resut_df_positive["protein_id"] = [list(result["protein_id"])[enum] for enum, x in enumerate(only_positive_predictions) if x == 1]
    resut_df_positive["prediction"] = [list(best_prediction)[enum] for enum, x in enumerate(only_positive_predictions) if x == 1]
    resut_df_positive["confidence"] = [list(confidence)[enum] for enum, x in enumerate(only_positive_predictions) if x == 1]
       
    resut_df_negative["protein_id"] = [list(result["protein_id"])[enum] for enum, x in enumerate(only_positive_predictions) if x == 0]
    resut_df_negative["prediction"] = [list(best_prediction)[enum] for enum, x in enumerate(only_positive_predictions) if x == 0]
    resut_df_negative["confidence"] = [list(confidence)[enum] for enum, x in enumerate(only_positive_predictions) if x == 0]
    

    """      
    save file
    """
    resut_df_positive.to_csv(path_or_buf=args.prediction_path  + "/" + "positive_prediction_summary"  + ".csv", mode = "a", header = True)
    resut_df_negative.to_csv(path_or_buf=args.prediction_path  + "/" + "negative_prediction_summary"  + ".csv", mode = "a", header = True)
