import json
from datetime import datetime
import torch.nn as nn
import torch


from args import get_parser
from utils import *
from mtad_gat import MTAD_GAT
from prediction import Predictor
from training import Trainer

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def print_gpu_info():
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"Memory Allocated: {torch.cuda.memory_allocated(i) / 1024**2:.2f} MB")
        print(f"Memory Cached: {torch.cuda.memory_reserved(i) / 1024**2:.2f} MB")

if __name__ == "__main__":

    id = datetime.now().strftime("%d%m%Y_%H%M%S")

    print_gpu_info()

    parser = get_parser()
    args = parser.parse_args() 

    dataset = args.dataset
    window_size = args.lookback
    spec_res = args.spec_res 
    normalize = args.normalize 
    n_epochs = args.epochs
    batch_size = args.bs
    init_lr = args.init_lr
    val_split = args.val_split # 0.1
    shuffle_dataset = args.shuffle_dataset # default=True
    use_cuda = args.use_cuda # default=True
    print_every = args.print_every # default=1
    log_tensorboard = args.log_tensorboard # default=True
    group_index = args.group[0] 
    index = args.group[2:] 
    gpu_number = args.gpu_number
    args_summary = str(args.__dict__) 
    print(args_summary)
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_number}"
   
    if dataset in [ 'SWaT', 'WADI', 'PSM']:
        output_path = f'output/{dataset}/'
        (x_train, _), (x_test, y_test) = get_data(dataset, normalize=normalize)
    else :
        raise Exception(f'Dataset "{dataset}" not available.')


    log_dir = f'{output_path}/logs'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    save_path = f"{output_path}/{id}"

    x_train = torch.from_numpy(x_train).float()
    x_test = torch.from_numpy(x_test).float()
    n_features = x_train.shape[1]

    plot_train = x_train
    target_dims = get_target_dims(dataset) 
    if target_dims is None: 
        out_dim = n_features
        print(f"Will forecast and reconstruct all {n_features} input features")
    elif type(target_dims) == int:
        print(f"Will forecast and reconstruct input feature: {target_dims}")
        out_dim = 1
    else:
        print(f"Will forecast and reconstruct input features: {target_dims}") 
        out_dim = len(target_dims)

    train_dataset = SlidingWindowDataset(x_train, window_size, target_dims)
    test_dataset = SlidingWindowDataset(x_test, window_size, target_dims) 
    train_loader, val_loader, test_loader = create_data_loaders(
        train_dataset, batch_size, val_split, shuffle_dataset, test_dataset=test_dataset
    )
    
    model = MTAD_GAT(
        n_features,
        window_size,
        out_dim,
        kernel_size=args.kernel_size,
        use_gatv2=args.use_gatv2,
        feat_gat_embed_dim=args.feat_gat_embed_dim,
        time_gat_embed_dim=args.time_gat_embed_dim,
        gru_n_layers=args.gru_n_layers,
        gru_hid_dim=args.gru_hid_dim,
        forecast_n_layers=args.fc_n_layers,
        forecast_hid_dim=args.fc_hid_dim,
        recon_n_layers=args.recon_n_layers,
        recon_hid_dim=args.recon_hid_dim,
        dropout=args.dropout,
        alpha=args.alpha
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.init_lr)   
    forecast_criterion = nn.MSELoss()   
    recon_criterion = nn.MSELoss()  

    torch.cuda.reset_peak_memory_stats()    
    print("Before running the model:")
    print(f"Max memory allocated: {torch.cuda.max_memory_allocated() / 1024 ** 2:.2f} MB")
    print(f"Max memory reserved: {torch.cuda.max_memory_reserved() / 1024 ** 2:.2f} MB")

    trainer = Trainer(
        model,
        optimizer,
        window_size,
        n_features,
        target_dims,
        n_epochs,
        batch_size,
        init_lr,
        forecast_criterion,
        recon_criterion,
        use_cuda,
        save_path,
        log_dir,
        print_every,
        log_tensorboard,
        args_summary
    )

    trainer.process_data["original_train"] = plot_train
    trainer.process_data["dataset"] = args.dataset
    trainer.fit(train_loader, val_loader) 

    test_loss = trainer.evaluate(test_loader)
    print(f"Test forecast loss: {test_loss[0]:.5f}")
    print(f"Test reconstruction loss: {test_loss[1]:.5f}")
    print(f"Test total loss: {test_loss[2]:.5f}")

 
    level_q_dict = {
        "SMAP": (0.75, 0.01),
        "MSL":   (0.90, 0.001),
        "SWaT": (0.9950, 0.001),
        "WADI": (0.5, 0.001),
        "PSM": (0.9950, 0.001) 
    }
    key = "SMD-2" if args.dataset == "SMD" else args.dataset
    level, q = level_q_dict[key]
    if args.level is not None:
        level = args.level
    if args.q is not None:
        q = args.q


    reg_level_dict = {"SMAP": 0, "MSL": 0, "SWaT":0,"WADI":0} 
    reg_level = reg_level_dict[key]

    trainer.load(f"{save_path}/model.pt")
    prediction_args = {
        'dataset': dataset,
        "target_dims": target_dims,
        'scale_scores': args.scale_scores,
        "level": level,
        "q": q,
        'dynamic_pot': args.dynamic_pot,
        "use_mov_av": args.use_mov_av,
        "gamma": args.gamma,
        "reg_level": reg_level,
        "save_path": save_path,
        "data_group": args.group,
    }
    best_model = trainer.model
    # 构建异常预测模型
    predictor = Predictor(
        best_model,
        window_size,
        n_features,
        prediction_args,
    )

    label = y_test[window_size:] if y_test is not None else None
    predictor.predict_anomalies(x_train, x_test, label,epochs=n_epochs)

    args_path = f"{save_path}/config.txt"
    with open(args_path, "w") as f:
        json.dump(args.__dict__, f, indent=2)
