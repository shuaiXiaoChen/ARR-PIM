import os
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler, dataset
from itertools import chain
import psutil
import seaborn as sns


def normalize_data(data, scaler=None):
    data = np.asarray(data, dtype=np.float32)
    if np.any(sum(np.isnan(data))): 
        data = np.nan_to_num(data)

    if scaler is None: 
        scaler = MinMaxScaler()
        scaler.fit(data) 
    data = scaler.transform(data) 
    print("Data normalized")

    return data, scaler 


def get_data_dim(dataset):

    if dataset == "SMAP":
        return 25
    elif dataset == "MSL":
        return 5
    elif dataset == "SWaT":
        return 51
    elif dataset == "WADI":
        return 123  
    else:
        raise ValueError("unknown dataset " + str(dataset))


def get_target_dims(dataset):

    if dataset == "SMAP":
        return None
    elif dataset == "MSL":
        return None
    elif dataset == "WADI":
        return None
    elif dataset == "SWaT":
        return None
    else:
        raise ValueError("unknown dataset " + str(dataset))


def get_data(dataset, max_train_size=None, max_test_size=None,
             normalize=False, spec_res=False, train_start=0, test_start=0):
    prefix = "datasets"
    flag = None
    if str(dataset).startswith("MSL"):
        flag = "MSL"
        prefix = prefix + "/SourceDatasets/MSL/" + dataset 
    elif str(dataset).startswith("SMAP"):
        flag = "SMAP"
        prefix = prefix + "/SourceDatasets/SMAP/" + dataset
    elif dataset in ["WADI", "SWaT"]:
        prefix = prefix + f"/SourceDatasets/{dataset}/{dataset}"
        flag = dataset

    if max_train_size is None:
        train_end = None
    else:
        train_end = train_start + max_train_size

    if max_test_size is None:
        test_end = None
    else:
        test_end = test_start + max_test_size
    print("load data of:", dataset)
    print("train: ", train_start, train_end)
    print("test: ", test_start, test_end)

    if flag:
        print("flag:",flag)
        x_dim = get_data_dim(flag)
        f = open(os.path.join(prefix+ "_train.pkl"), "rb")  
    else:
        x_dim = get_target_dims(dataset)
        f = open(os.path.join(prefix, dataset + "_train.pkl"), "rb")
    train_data = pickle.load(f).reshape((-1, x_dim))[train_start:train_end, :] 
    print("hello ,train load well and to numpy and reshape and iloc!!!")
    print("train_data.shape: ", train_data.shape)
    f.close()

    try:
        if flag:
            f = open(os.path.join(prefix+ "_test.pkl"), "rb")
            test_data = pickle.load(f).reshape((-1, x_dim))[test_start:test_end, :]
            print("test_data.shape:", test_data.shape)
            f.close()
        else:
            f = open(os.path.join(prefix, dataset + "_test.pkl"), "rb")
            test_data = pickle.load(f).reshape((-1, x_dim))[test_start:test_end, :]
            print("test_data.shape:", test_data.shape)
            f.close()
    except (KeyError, FileNotFoundError):
        test_data = None

    try:
        if flag:
            f = open(os.path.join(prefix + "_test_label.pkl"), "rb")
            test_label = pickle.load(f).reshape((-1))[test_start:test_end]
            print("test_label.shape: ", test_label.shape)
            f.close()
        else:
            f = open(os.path.join(prefix, dataset + "_test_label.pkl"), "rb")
            test_label = pickle.load(f).reshape((-1))[test_start:test_end]
            print("test_label.shape: ", test_label.shape)
            f.close()
    except (KeyError, FileNotFoundError):
        test_label = None

    if normalize:
        train_data, scaler = normalize_data(train_data, scaler=None) 
        test_data, _ = normalize_data(test_data, scaler=scaler) 
    print("train set shape: ", train_data.shape)
    print("test set shape: ", test_data.shape)
    print("test set label shape: ", None if test_label is None else test_label.shape)
    return (train_data, None), (test_data, test_label)


def load_MSL_SMAP(dataset):
    train_data = dataset["train_data"]
    test_data = dataset["test_data"]
    test_label = dataset["test_label"]
    lenth1 = len(train_data)
    lenth2 = len(test_data)
    lenth3 = len(test_label)
    for i in range(lenth1):
        pass
    for i in range(lenth2):
        pass
    for i in range(lenth3):
        pass
class SlidingWindowDataset(Dataset): 
    def __init__(self, data, window, target_dim=None, horizon=1):
        self.data = data
        self.window = window
        self.target_dim = target_dim
        self.horizon = horizon

    def __getitem__(self, index):
        x = self.data[index : index + self.window]
        y = self.data[index + self.window : index + self.window + self.horizon]
        return x, y

    def __len__(self):
        return len(self.data) - self.window 


def create_data_loaders(train_dataset, batch_size, val_split=0.1, shuffle=True, test_dataset=None):
    train_loader, val_loader, test_loader = None, None, None
    if val_split == 0.0:
        print(f"train_size: {len(train_dataset)}")
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle,drop_last=True)

    else:
        dataset_size = len(train_dataset)
        indices = list(range(dataset_size)) 
        split = int(np.floor(val_split * dataset_size)) 
        if shuffle:
            np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]

        train_sampler = SubsetRandomSampler(train_indices) 
        valid_sampler = SubsetRandomSampler(val_indices) 

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, drop_last=True)
        val_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=valid_sampler, drop_last=True)

        print(f"train_size: {len(train_indices)}")
        print(f"validation_size: {len(val_indices)}")

    if test_dataset is not None:
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,drop_last=True)
        print(f"test_size: {len(test_dataset)}")

    return train_loader, val_loader, test_loader


def plot_losses(losses, save_path="", plot=True): 
    """
    :param losses: dict with losses
    :param save_path: path where plots get saved
    """

    plt.plot(losses["train_forecast"], label="Forecast loss")
    plt.plot(losses["train_recon"], label="Recon loss")
    plt.plot(losses["train_total"], label="Total loss")
    plt.title("Training losses during training")
    plt.xlabel("Epoch")
    plt.ylabel("RMSE")
    plt.legend()
    plt.savefig(f"{save_path}/train_losses.png", bbox_inches="tight")
    if plot:
        plt.show()
    plt.close()

    plt.plot(losses["val_forecast"], label="Forecast loss")
    plt.plot(losses["val_recon"], label="Recon loss")
    plt.plot(losses["val_total"], label="Total loss")
    plt.title("Validation losses during training")
    plt.xlabel("Epoch")
    plt.ylabel("RMSE")
    plt.legend()
    plt.savefig(f"{save_path}/validation_losses.png", bbox_inches="tight")
    if plot:
        plt.show()
    plt.close()

def convert_tensor_to_list(tensor_or_list):
    if isinstance(tensor_or_list, torch.Tensor):
        return tensor_or_list.tolist()
    elif isinstance(tensor_or_list, list):
        return [convert_tensor_to_list(item) for item in tensor_or_list]
    else:
        return tensor_or_list

def convert_tensor_to_list_1(tensor_or_list):
    print("to convert_tensor_to_list_1")
    tensor_or_list = [[[[[item_alone.item()] for item_alone in column] for column in row] for row in e_p] for e_p in tensor_or_list]
    print("len(tensor_or_list):", len(tensor_or_list),
          "type(tensor_or_list):", type(tensor_or_list),
          "len(tensor_or_list[0]",len(tensor_or_list[0]))
    print("tensor_or_list[epoch=0][column=1][row=1]:", tensor_or_list[0][1][1])
    return tensor_or_list
def tensor_to_list_generator(tensor):
    if isinstance(tensor, torch.Tensor):
        yield tensor.item()
    elif isinstance(tensor, list):
        for item in tensor:
            yield from tensor_to_list_generator(item)
def plot_data(process_data, save_path="", plot=True):  
    """
    :param losses: dict with data
    :param save_path: path where plots get saved
    """
    def convert_to_numpy(tensor_data):
        if isinstance(tensor_data, list):
            tensor_data = torch.stack([torch.tensor(item) if isinstance(item, list) else item for item in tensor_data])
        if not tensor_data.is_cuda:
            return tensor_data.cpu().numpy()
        else:
            return tensor_data.cpu().numpy()


    print("x_x[0]: ", len(process_data["x_x"][0]))
    if process_data["dataset"] =="SMD":
        x_x_list = []
        x_recon_list = []
        for x_tensor in process_data["x_x"]:
            if x_tensor:
                x_x_list.append(convert_to_numpy(x_tensor).squeeze())
        for x_tensor in process_data["x_recon"]:
            if x_tensor:
                x_recon_list.append(convert_to_numpy(x_tensor).squeeze())
    else:
        x_x_list = [convert_to_numpy(x_tensor).squeeze() for x_tensor in process_data["x_x"] if x_tensor]
        x_recon_list = [convert_to_numpy(x_tensor).squeeze() for x_tensor in process_data["x_recon"] if x_tensor]
    y_y_list = [convert_to_numpy(y_tensor) for y_tensor in process_data["y_y"] if y_tensor]
    y_pred_list = [convert_to_numpy(y_tensor) for y_tensor in process_data["y_pred"] if y_tensor]


    for i in range(len(x_x_list)):
        print(f"x_x[{i}].shape:", x_x_list[i].shape)
        print(f"x_recon[{i}].shape:", x_recon_list[i].shape)
        print(f"y_y[{i}].shape:", y_y_list[i].shape)
        print(f"y_pred[{i}].shape:", y_pred_list[i].shape)


    epoch_num = len(x_x_list)

    x_x_heap_map = [x_x_list[i] for i in range(epoch_num)]
    x_recon_heap_map = [x_recon_list[i] for i in range(epoch_num)]

    x_x_list_a = [x_x_list[i][:, 2] for i in range(epoch_num)]    
    x_recon_list_a = [x_recon_list[i][:, 2] for i in range(epoch_num)]   

    for i in range(epoch_num):
        plt.plot(x_recon_list_a[i][:100], label="recon_data")  
        plt.plot(x_x_list_a[i][:100], label="orig_train")
        plt.title("reconstruct and predict data")
        plt.xlabel("Time")
        plt.ylabel("Data")
        plt.legend(loc='upper right')
        plt.savefig(f"{save_path}/train_data_x_{i}.png", bbox_inches="tight")
        if plot:
            plt.show()
        plt.close()
    for i in range(epoch_num):
        plt.plot(y_pred_list[i][:100], label="pred_data")  
        plt.plot(y_y_list[i][:100], label="label_data")
        plt.title("predict")
        plt.xlabel("Time")
        plt.ylabel("Data")
        plt.legend()
        plt.savefig(f"{save_path}/train_data_y_{i}.png", bbox_inches="tight")
        if plot:
            plt.show()
        plt.close()

        
    for i in range(epoch_num):
     
        x_x_2d_mean = np.mean(x_x_list[i], axis=1)  
        x_recon_2d_mean = np.mean(x_recon_list[i], axis=1) 

        x_x_2d_max = np.max(x_x_list[i], axis=1)  
        x_recon_2d_max = np.max(x_recon_list[i], axis=1) 

        x_x_heap_map_df_mean = pd.DataFrame(x_x_2d_mean)
        x_recon_heap_map_df_mean = pd.DataFrame(x_recon_2d_mean)

        x_x_heap_map_df_max = pd.DataFrame(x_x_2d_max)
        x_recon_heap_map_df_max = pd.DataFrame(x_recon_2d_max)

        corr_matrix_x_x_mean = x_x_heap_map_df_mean.corr()
        corr_matrix_x_recon_mean = x_recon_heap_map_df_mean.corr()

        corr_matrix_x_x_max = x_x_heap_map_df_max.corr()
        corr_matrix_x_recon_max = x_recon_heap_map_df_max.corr()

  
        sns.heatmap(corr_matrix_x_x_mean, annot=False, cmap='coolwarm', center=0)
        plt.title(f'Correlation between Variables in Original Data (Mean)')

        plt.savefig(f"{save_path}/corr_matrix_x_mean_{i}.png", bbox_inches="tight")
        plt.show() if plot else plt.close()

        sns.heatmap(corr_matrix_x_recon_mean, annot=False, cmap='coolwarm', center=0)
        plt.title(f'Correlation between Variables in Reconstructed Data (Mean)')

        plt.savefig(f"{save_path}/corr_matrix_recon_mean_{i}.png", bbox_inches="tight")
        plt.show() if plot else plt.close()

        sns.heatmap(corr_matrix_x_x_max, annot=False, cmap='coolwarm', center=0)
        plt.title(f'Correlation between Variables in Original Data (Max)')

        plt.savefig(f"{save_path}/corr_matrix_x_max_{i}.png", bbox_inches="tight")
        plt.show() if plot else plt.close()

        sns.heatmap(corr_matrix_x_recon_max, annot=False, cmap='coolwarm', center=0)
        plt.title(f'Correlation between Variables in Reconstructed Data (Max)')

        plt.savefig(f"{save_path}/corr_matrix_recon_max_{i}.png", bbox_inches="tight")
        plt.show() if plot else plt.close()


def load(model, PATH, device="cpu"):
    """
    Loads the model's parameters from the path mentioned
    :param PATH: Should contain pickle file
    """
    model.load_state_dict(torch.load(PATH, map_location=device))


def get_series_color(y):
    if np.average(y) >= 0.95:
        return "black"
    elif np.average(y) == 0.0:
        return "black"
    else:
        return "black"


def get_y_height(y):
    if np.average(y) >= 0.95:
        return 1.5
    elif np.average(y) == 0.0:
        return 0.1
    else:
        return max(y) + 0.1


def adjust_anomaly_scores(scores, dataset, is_train, lookback):
    """
    Method for MSL and SMAP where channels have been concatenated as part of the preprocessing
    :param scores: anomaly_scores
    :param dataset: name of dataset
    :param is_train: if scores is from train set
    :param lookback: lookback (window size) used in model
    """


    if dataset.upper() not in ['SMAP', 'MSL']:
        return scores

    adjusted_scores = scores.copy()
    if is_train:
        md = pd.read_csv(f'./datasets/data/{dataset.lower()}_train_md.csv')
    else:
        md = pd.read_csv('./datasets/data/labeled_anomalies.csv')
        md = md[md['spacecraft'] == dataset.upper()]

    md = md[md['chan_id'] != 'P-2']


    md = md.sort_values(by=['chan_id'])


    sep_cuma = np.cumsum(md['num_values'].values) - lookback
    sep_cuma = sep_cuma[:-1]
    buffer = np.arange(1, 20)
    i_remov = np.sort(np.concatenate((sep_cuma, np.array([i+buffer for i in sep_cuma]).flatten(),
                                      np.array([i-buffer for i in sep_cuma]).flatten())))
    i_remov = i_remov[(i_remov < len(adjusted_scores)) & (i_remov >= 0)]
    i_remov = np.sort(np.unique(i_remov))
    if len(i_remov) != 0:
        adjusted_scores[i_remov] = 0

    # Normalize each concatenated part individually
    sep_cuma = np.cumsum(md['num_values'].values) - lookback
    s = [0] + sep_cuma.tolist()
    for c_start, c_end in [(s[i], s[i+1]) for i in range(len(s)-1)]:
        e_s = adjusted_scores[c_start: c_end+1]

        if e_s.size == 0: 
                        
            continue  

        e_s = (e_s - np.min(e_s))/(np.max(e_s) - np.min(e_s))
        adjusted_scores[c_start: c_end+1] = e_s

    return adjusted_scores
