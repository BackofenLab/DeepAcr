
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
import glob as glob
import pathlib

def isBetween(value, intervall):


    return intervall[1] < value <= intervall[0]
    
    
def create_databse_file(path = "./"):



    
    ke = df.keys()

    
    pred1 = df["ensamble_LSTM"]
    pred2 = df["ensamble_GRU"]
    pred3 = df["ensamble_LINEAR"]
    ID = df["protein_id"]
  
    
    pos_list = []
    neg_list = []
    
    column1 = ["ID", "model_prediction", "protein_class", "confidence"]
    column2 = ["ID", "model_prediction", "protein_class", "confidence"]
    
    for enum in range(len(pred1)):


        print(path)
       
        if int(pred1[enum]) == 1 or int(pred2[enum]) == 1 or int(pred3[enum]) == 1:
            pos_LSTM = df["LSTM_positive_prediction"]
            pos_GRU = df["GRU_positive_prediction"]
            pos_LINEAR = df["LINEAR_positive_prediction"]
            
            max_ = np.max([float(pos_LSTM[enum]), float(pos_GRU[enum]), float(pos_LINEAR[enum])])
            
            
            conf_dict = {"high":[1.01, 0.80] , "medium": [0.80, 0.65], "low": [0.65, 0.50]}
    
            conf = [k for k in list(conf_dict.keys()) if isBetween(max_, conf_dict[k])]          
            pos_list.append([ID[enum], max_, 1, conf[0]])
            
        
        else:
    
            neg_LSTM = df["LSTM_negative_prediction"]
            neg_GRU = df["GRU_negative_prediction"]
            neg_LINEAR = df["LINEAR_negative_prediction"]
            

            
            
            max_ = np.max([float(neg_LSTM[enum]), float(neg_GRU[enum]), float(neg_LINEAR[enum])])
            conf_dict = {"high":[1.01, 0.80] , "medium": [0.80, 0.65], "low": [0.65, 0.50]}

            conf = [k for k in list(conf_dict.keys()) if isBetween(max_, conf_dict[k])]

            neg_list.append([ID[enum], max_, 0,  conf[0]])
            
            
    df_pos = pd.DataFrame(pos_list, columns = column1)
    df_neg = pd.DataFrame(neg_list, columns = column2)
    
    df_pos.to_csv(path + "positive_prediction.csv")
    df_neg.to_csv(path + "negative_prediction.csv")
      


if __name__ == "__main__":


    folders = glob.glob("./models/*")

    
    for folder in folders:


        d_path1 = folder + "/with_infos/"
        d_path2 = folder + "/without_infos/"
    

        create_databse_file(path = d_path1)
        create_databse_file(path = d_path2)



