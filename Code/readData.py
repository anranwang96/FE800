import pandas as pd
import numpy as np
import glob, os
class Read_data():
    
    #Read all data
    def read_all_data(self, path):
        file_name = glob.glob(os.path.join(path, "*.csv"))
        all_data = []
        for f in file_name:
            all_data.append(pd.read_csv(f))
        return all_data, file_name
    
    #Get data index
    def data_index(self, file_name):
        data_index = []
        for i in range(len(file_name)):
            if file_name[i][-17] == '/':
                data_index.append(file_name[i][-16:-14])
            elif file_name[i][-18] == '/':
                data_index.append(file_name[i][-17:-14])
            elif file_name[i][-16] == '/':
                data_index.append(file_name[i][-15])
            elif file_name[i][-20] == '/':
                data_index.append(file_name[i][-19:-14])
            else: 
                data_index.append(file_name[i][-18:-14])
        return data_index
    
    #Close price dataset
    def dataset(self, all_data, data_index):
        length = np.asarray([len(all_data[i]) for i in range(len(all_data))])
        length_max = np.max(length)

        close_dataset = pd.DataFrame(np.nan, index = all_data[np.where(length == length_max)[0][0]]['Date'], columns = data_index)

        for i in range(len(data_index)):
            if len(all_data[i]) < length_max:
                close_dataset.iloc[length_max-len(all_data[i]):, i] = list(all_data[i]['Close'])
            else:
                close_dataset.iloc[:, i] = list(all_data[i]['Close'])
        return close_dataset
