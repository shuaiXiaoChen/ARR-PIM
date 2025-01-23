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

    id = datetime.now().strftime("%d%m%Y_%H%M%S") # id以时间命名

    print_gpu_info()

    parser = get_parser() # 创建实例化对象
    args = parser.parse_args() # 解析parser的参数

    dataset = args.dataset
    window_size = args.lookback # 设定窗口大小
    spec_res = args.spec_res # 这个参数用途没搞清楚
    normalize = args.normalize # 归一化
    n_epochs = args.epochs
    batch_size = args.bs
    init_lr = args.init_lr
    val_split = args.val_split # 0.1
    shuffle_dataset = args.shuffle_dataset # default=True
    use_cuda = args.use_cuda # default=True
    print_every = args.print_every # default=1
    log_tensorboard = args.log_tensorboard # default=True
    group_index = args.group[0] # default="1-1", help="Required for SMD dataset. <group_index>-<index>",第i个机子
    index = args.group[2:] # default="1-1", help="Required for SMD dataset. <group_index>-<index>" 第i个机子的第j个传感器
    gpu_number = args.gpu_number
    args_summary = str(args.__dict__) # 将参数转化为字典（二者可以互转）
    print(args_summary)
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_number}"
    # 这是分机器的统一实现
    # if dataset == 'SMD': # 默认SMD
    #     output_path = f'output/SMD/{args.group}' # default="1-1"：output/SMD/1-1
    #     (x_train, _), (x_test, y_test) = get_data(f"machine-{group_index}-{index}", normalize=normalize)    # 分机器获取数据集和预处理
    # elif dataset in ['MSL', 'SMAP']:
    #     output_path = f'output/{dataset}/{args.group}' # default = "group:MSL/SMAP A-1:T-13"  这个代码用于分机器处理
    #     (x_train, _), (x_test, y_test) = get_data(f"{dataset}/{group_index}-{index}", normalize=normalize)   # 这个代码用于分机器处理
    # elif dataset in ['SWaT','WADI','PSM']:
    #     output_path = f'output/{dataset}/'
    #     (x_train, _), (x_test, y_test) = get_data(dataset, normalize=normalize)
    # else:
    #     raise Exception(f'Dataset "{dataset}" not available.')

    # 这是不分机器的统一实现
    if dataset in ['SMD', 'SWaT', 'WADI', 'PSM','MSL','SMAP']:
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
    target_dims = get_target_dims(dataset) # 返回[0]
    if target_dims is None: # dataset == "SMD"
        out_dim = n_features
        print(f"Will forecast and reconstruct all {n_features} input features")
    elif type(target_dims) == int: # 整型数据重构?[0],预设值的值
        print(f"Will forecast and reconstruct input feature: {target_dims}")
        out_dim = 1
    else:
        print(f"Will forecast and reconstruct input features: {target_dims}") # 重构什么呢?这个维度dim怎么确定？
        out_dim = len(target_dims)

    # 获取训练数据集和测试数据集
    train_dataset = SlidingWindowDataset(x_train, window_size, target_dims) # 这里不是返回了两个吗
    test_dataset = SlidingWindowDataset(x_test, window_size, target_dims) # 理解了，一个是窗口，一个是预测值

    # 创建数据集迭代器
    train_loader, val_loader, test_loader = create_data_loaders(
        train_dataset, batch_size, val_split, shuffle_dataset, test_dataset=test_dataset
    ) # 加载数据集

    # 初始化模型，返回预测值和重构值 return predictions：y, recons：x
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

    optimizer = torch.optim.Adam(model.parameters(), lr=args.init_lr)   # 定义优化器
    forecast_criterion = nn.MSELoss()   # 预测损失函数
    recon_criterion = nn.MSELoss()  # 重构损失函数(这里可以考虑换一个重构损失KL散度)

    torch.cuda.reset_peak_memory_stats()    # 查看cuda使用情况
    print("Before running the model:")
    print(f"Max memory allocated: {torch.cuda.max_memory_allocated() / 1024 ** 2:.2f} MB")
    print(f"Max memory reserved: {torch.cuda.max_memory_reserved() / 1024 ** 2:.2f} MB")

    # 开始训练
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
    trainer.fit(train_loader, val_loader) # 这是拟合，其实就是训练环节
    # 下面的三行用于可视化分析
    # plot_losses(trainer.losses, save_path=save_path, plot=False) # 损失函数可视化
    # plot_data(trainer.process_data,save_path=save_path, plot=False) # 重构数据可视化

    # Check test loss
    test_loss = trainer.evaluate(test_loader) # 测试损失及评估
    print(f"Test forecast loss: {test_loss[0]:.5f}")
    print(f"Test reconstruction loss: {test_loss[1]:.5f}")
    print(f"Test total loss: {test_loss[2]:.5f}")

    # Some suggestions for POT args # 阈值设置
    level_q_dict = {
        # "SMAP": (0.70, 0.005),
        # "SMAP": (0.70, 0.001),
        "SMAP": (0.75, 0.01),
        "MSL":   (0.90, 0.001),
        # "MSL": (0.95, 0.001),
        "SMD-1": (0.9950, 0.001),
        # "SMD-2": (0.9925, 0.001),
        "SMD-2": (0.9925, 0.0001),
        "SMD-3": (0.9999, 0.001),
        "SWaT": (0.9950, 0.001),
        # "WADI": (0.84, 0.0001),
        "WADI": (0.5, 0.001),
        # "WADI": (0.99, 0.001),
        # "WADI": (0.995, 0.001),
        "PSM": (0.9950, 0.001)  # 默认值
    } # 设置一个字典for-q
    # key = "SMD-" + args.group[0] if args.dataset == "SMD" else args.dataset # 查字典的key
    key = "SMD-2" if args.dataset == "SMD" else args.dataset # 查字典的key
    level, q = level_q_dict[key] # 查字典，获取元组
    if args.level is not None:
        level = args.level
    if args.q is not None:
        q = args.q

    # Some suggestions for Epsilon args
    reg_level_dict = {"SMAP": 0, "MSL": 0, "SMD-1": 1, "SMD-2": 1,
                      "SMD-3": 1,"SWaT":0,"WADI":0,"PSM":0} # 怎么又设置一个字典
    # key = "SMD-" + args.group[0] if dataset == "SMD" else dataset # 又设置了一个字典for：level
    key = "SMD-2" if dataset == "SMD" else dataset # 又设置了一个字典for：level
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

    # Save config
    args_path = f"{save_path}/config.txt"
    with open(args_path, "w") as f:
        json.dump(args.__dict__, f, indent=2)

    # plot_data(trainer.process_data, save_path=save_path, plot=False)