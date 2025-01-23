import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


class Trainer:
    """Trainer class for MTAD-GAT model.

    :param model: MTAD-GAT model
    :param optimizer: Optimizer used to minimize the loss function
    :param window_size: Length of the input sequence
    :param n_features: Number of input features
    :param target_dims: dimension of input features to forecast and reconstruct
    :param n_epochs: Number of iterations/epochs
    :param batch_size: Number of windows in a single batch
    :param init_lr: Initial learning rate of the module
    :param forecast_criterion: Loss to be used for forecasting.
    :param recon_criterion: Loss to be used for reconstruction.
    :param boolean use_cuda: To be run on GPU or not
    :param dload: Download directory where models are to be dumped
    :param log_dir: Directory where SummaryWriter logs are written to
    :param print_every: At what epoch interval to print losses
    :param log_tensorboard: Whether to log loss++ to tensorboard
    :param args_summary: Summary of args that will also be written to tensorboard if log_tensorboard
    """

    def __init__(
        self,
        model,
        optimizer,
        window_size,
        n_features,
        target_dims=None,
        n_epochs=200,
        batch_size=256,
        init_lr=0.001,
        forecast_criterion=nn.MSELoss(),
        recon_criterion=nn.MSELoss(),
        use_cuda=True,
        dload="",
        log_dir="output/",
        print_every=1, # 多少间隔打印损失
        log_tensorboard=True,
        args_summary="",
    ):

        self.model = model
        self.optimizer = optimizer
        self.window_size = window_size
        self.n_features = n_features
        self.target_dims = target_dims
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.init_lr = init_lr
        self.forecast_criterion = forecast_criterion
        self.recon_criterion = recon_criterion
        # self.device = "cuda" if use_cuda and torch.cuda.is_available() else "cpu"
        self.device = "cuda" if use_cuda and torch.cuda.is_available() else "cpu"
        self.dload = dload
        self.log_dir = log_dir
        self.print_every = print_every
        self.log_tensorboard = log_tensorboard

        self.losses = {
            "train_total": [],
            "train_forecast": [],
            "train_recon": [],
            "val_total": [],
            "val_forecast": [],
            "val_recon": [],
        }
        self.process_data = {
            "y_y": [],
            "y_pred": [],
            "x_x": [],
            "x_recon": []
        }
        self.epoch_times = []
        # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
        # if self.device == "cuda":
        if self.device == "cuda":
            # print("if self.device == cuda:(of def __init__ in training.py)",self.device)
            self.model.cuda()

        if self.log_tensorboard:
            self.writer = SummaryWriter(f"{log_dir}")
            self.writer.add_text("args_summary", args_summary)

    def fit(self, train_loader, val_loader=None):
        """Train model for self.n_epochs.
        Train and validation (if validation loader given) losses stored in self.losses

        :param train_loader: train loader of input data
        :param val_loader: validation loader of input data
        """

        init_train_loss = self.evaluate(train_loader) # 评估验证损失 return forecast_loss, recon_loss, total_loss
        print(f"Init total train loss: {init_train_loss[2]:5f}") # 打印训练损失

        if val_loader is not None:
            init_val_loss = self.evaluate(val_loader) # 评估验证数据集
            print(f"Init total val loss: {init_val_loss[2]:.5f}") # 打印验证损失

        # 开始训练epochs次
        print(f"Training model for {self.n_epochs} epochs..")
        train_start = time.time() # 记录训练时间
        for epoch in range(self.n_epochs):
            epoch_start = time.time() # 记录每个epoch开始时间
            self.model.train() # 进入训练模式
            forecast_b_losses = [] # 存储每一个batch的预测损失
            recon_b_losses = [] # 存储每一个batch的重构损失\
            y_y = []
            y_pred = []
            x_x = []
            x_recon = []

            # print(f"code run in training.py and Current device: {self.device}")
            pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{self.n_epochs}', unit='batch')  # # 2024.7.29添加

            # for x, y in train_loader: # x：数据集，y：标签，迭代数据
            for x, y in pbar:   # 2024.7.29添加
                # print("self.device(of def fit in training.py):",self.device)
                x = x.to(self.device)
                y = y.to(self.device)
                self.optimizer.zero_grad() # 优化器梯度清零

                preds, recons = self.model(x) # 模型训练有两个输出：预测、重构 return predictions, recons
                # self.process_data["plot_recon"].append(recons)
                # self.process_data["plot_forecast"].append(preds)

                if self.target_dims is not None: # 在第三个维度上进行切片
                    x = x[:, :, self.target_dims]
                    y = y[:, :, self.target_dims].squeeze(-1) # 如果最后一个维度是1，移除


                if preds.ndim == 3:
                    preds = preds.squeeze(1) # 如果第二个维度是1，移除，不是1就不移除

                if y.ndim == 3:
                    y = y.squeeze(1)


                # if self.process_data["dataset"] != "SMD" and self.process_data["dataset"] != "WADI" \
                #         and epoch == (self.n_epochs-1):
                #     x_x.extend(x.detach().cpu())
                #     x_recon.extend(recons.detach().cpu())  # x_x[batch_size][clow=len][clown=38]
                #     y_y.extend(y.detach())
                #     y_pred.extend(preds.detach())
                # elif self.process_data["dataset"] == "WADI" and epoch == (self.n_epochs-1):
                #     # elif self.process_data["dataset"] == "WADI" and epoch in [0, 2, 4, 6, 8, 9]:
                #     x_x1 = x[0:200]
                #     x_recon1 = recons[0:200]
                #     # x_x.extend(x_x1.detach().cpu())
                #     x_x1 = None
                #     # x_recon.extend(x_recon1.detach().cpu())  # x_x[batch_size][clow=len][clown=38]
                #     x_recon1 = None
                #     y_y.extend(y.detach())
                #     y_pred.extend(preds.detach())
                # elif self.process_data["dataset"] == "SMD" and epoch == (self.n_epochs-1):
                #     x_x1 = x[0:300]
                #     x_recon1 = recons[0:300]
                #     x_x.extend(x_x1.detach().cpu())
                #     x_x1 = None
                #     x_recon.extend(x_recon1.detach().cpu())   # x_x[batch_size][clow=len][clown=38]
                #     x_recon1 = None
                # #     y_y.extend(y.detach())
                #     y_pred.extend(preds.detach())


                forecast_loss = torch.sqrt(self.forecast_criterion(y, preds)) # 使用均方误差损失
                recon_loss = torch.sqrt(self.recon_criterion(x, recons))
                loss = forecast_loss + recon_loss # 总损失，用于计算梯度和优化参数
                pbar.set_postfix(loss=loss / len(train_loader))     # 2024.7.29添加

                loss.backward() # 反向传播、计算梯度
                self.optimizer.step() # 更新参数

                forecast_b_losses.append(forecast_loss.item()) # 记录每一batch预测损失
                recon_b_losses.append(recon_loss.item()) # 记录每一batch重构损失


            # torch.cuda.empty_cache()
            # print("After running the model:")
            # print(f"Max memory allocated: {torch.cuda.max_memory_allocated() / 1024 ** 2:.2f} MB")
            # print(f"Max memory reserved: {torch.cuda.max_memory_reserved() / 1024 ** 2:.2f} MB")

            print("traing.py")
            #
            # self.process_data["y_y"].append(y_y)
            # y_y = None
            # self.process_data["y_pred"].append(y_pred)
            # y_pred = None
            # self.process_data["x_x"].append(x_x)
            # x_x = None
            # self.process_data["x_recon"].append(x_recon)
            # x_recon = None
            # print("self.process_data 的长度，判定是否有数据")
            # print(len(self.process_data["y_y"]),len(self.process_data["y_pred"]),
            #       len(self.process_data["x_x"]),len(self.process_data["x_recon"]))
            # print("len(x_x)", len(x_x), "x_x", x_x[0][:10])     # x[dim0=38][dim1=15537]

            forecast_b_losses = np.array(forecast_b_losses) # 列表转为np数组
            recon_b_losses = np.array(recon_b_losses)

            forecast_epoch_loss = np.sqrt((forecast_b_losses ** 2).mean()) # 预测总误差求均值
            recon_epoch_loss = np.sqrt((recon_b_losses ** 2).mean()) # 重构总误差求均值

            total_epoch_loss = forecast_epoch_loss + recon_epoch_loss # 总误差=均值预测误差+均值重构误差

            # 在列表中操作可以方便形成图表
            self.losses["train_forecast"].append(forecast_epoch_loss) # 存储每一个epoch的预测总均值损失
            self.losses["train_recon"].append(recon_epoch_loss) # 存储每一个epoch的重构总均值损失
            self.losses["train_total"].append(total_epoch_loss) # 存储总损失（重构+预测）

            # Evaluate on validation set 在验证集上进行评估
            forecast_val_loss, recon_val_loss, total_val_loss = "NA", "NA", "NA"
            if val_loader is not None: # 存储中的验证损失：预测验证损失、重构验证损失、总验证损失
                forecast_val_loss, recon_val_loss, total_val_loss = self.evaluate(val_loader)
                self.losses["val_forecast"].append(forecast_val_loss)
                self.losses["val_recon"].append(recon_val_loss)
                self.losses["val_total"].append(total_val_loss)
                # 如果最新验证总损失<=验证损失列表的最后一个，保留模型，这个if应该往前移，先比较，然后再添加列表
                if total_val_loss <= self.losses["val_total"][-1]: # 早停（early stopping）策略，防止过拟合
                    self.save(f"model.pt")

            if self.log_tensorboard: # 保存结果
                self.write_loss(epoch)

            epoch_time = time.time() - epoch_start # 单次训练时长
            self.epoch_times.append(epoch_time)

            if epoch % self.print_every == 0: # 因为print_every=1，所以每个epoch都打印每一次迭代的预测、重构和总损失
                s = (
                    f"[Epoch {epoch + 1}] "
                    f"forecast_loss = {forecast_epoch_loss:.5f}, "
                    f"recon_loss = {recon_epoch_loss:.5f}, "
                    f"total_loss = {total_epoch_loss:.5f}"
                )

                if val_loader is not None: # 如果验证集不为空，同时打印验证集的损失
                    s += (
                        f" ---- val_forecast_loss = {forecast_val_loss:.5f}, "
                        f"val_recon_loss = {recon_val_loss:.5f}, "
                        f"val_total_loss = {total_val_loss:.5f}"
                    )

                s += f" [{epoch_time:.1f}s]"
                print(s) # 学会了，打印的新方法，先存着，之后一块打印

        if val_loader is None: # 如果验证集不为空，保留较优模型
            self.save(f"model.pt")

        train_time = int(time.time() - train_start) # 总训练时长
        if self.log_tensorboard: # log和打印版训练时长
            self.writer.add_text("total_train_time", str(train_time))
        print(f"-- Training done in {train_time}s.")

    def evaluate(self, data_loader):    # 评估或者说训练过程
        """Evaluate model

        :param data_loader: data loader of input data
        :return forecasting loss, reconstruction loss, total loss
        """

        self.model.eval() # 进入评估模式

        forecast_losses = []
        recon_losses = []

        with torch.no_grad():   # 上下文中，禁用梯度计算，以减少内存消耗和提高评估速度
            # 2024.7.29添加
            # pbar = tqdm(data_loader, desc=f'Evaluate model and return forecasting loss, reconstruction loss, total loss', unit='batch')
            pbar = tqdm(data_loader, unit='batch')
            # for x, y in data_loader: # x是特征，y是标签
            for x, y in pbar:   # 2024.7.29添加
                # print("self.device(of def evaluate in training.py):",self.device)
                x = x.to(self.device)
                y = y.to(self.device)

                preds, recons = self.model(x)   # 进行模型运算

                if self.target_dims is not None:    # 输入数据的维度，对加载的数据进行切片？怎么进行切片？
                    # print("x.shape and y.shape in training -evaluate:", x.shape,y.shape)
                    x = x[:, :, self.target_dims]   # 选择第三个维度（通常是特征维度）中的某些特定切片。
                    y = y[:, :, self.target_dims].squeeze(-1)   # 同样切片，并移除大小为 1 的最后一个维度。

                if preds.ndim == 3:     # 如果预测结果和目标数据是三维的（批次大小，时间步长，特征），则移除大小为 1 的第二个维度。
                    preds = preds.squeeze(1)
                if y.ndim == 3:
                    y = y.squeeze(1)

                # print("y.shape, preds.shape:",y.shape, preds.shape)
                forecast_loss = torch.sqrt(self.forecast_criterion(y, preds))   # 预测值求均方误差损失
                recon_loss = torch.sqrt(self.recon_criterion(x, recons))    # 重构值求均方误差损失

                forecast_losses.append(forecast_loss.item())    # 预测误差损失加入列表
                recon_losses.append(recon_loss.item())  # 重构误差损失加入列表
                pbar.set_postfix(loss=forecast_loss+recon_loss)     # 2024.7.29添加

        forecast_losses = np.array(forecast_losses)     # 预测误差损失转为np数组
        recon_losses = np.array(recon_losses)   # 重构误差损失转为np数组

        forecast_loss = np.sqrt((forecast_losses ** 2).mean())  # 所有预测误差求均值
        recon_loss = np.sqrt((recon_losses ** 2).mean())    # 所有重构误差求均值

        total_loss = forecast_loss + recon_loss     # 总误差损失=预测误差+重构误差

        return forecast_loss, recon_loss, total_loss    # 返回预测平均误差、重构平均误差、总误差

    def save(self, file_name):
        """
        Pickles the model parameters to be retrieved later
        :param file_name: the filename to be saved as,`dload` serves as the download directory
        """
        PATH = self.dload + "/" + file_name
        if os.path.exists(self.dload):
            pass
        else:
            os.mkdir(self.dload)
        torch.save(self.model.state_dict(), PATH)

    def load(self, PATH):
        """
        Loads the model's parameters from the path mentioned
        :param PATH: Should contain pickle file
        """
        self.model.load_state_dict(torch.load(PATH, map_location=self.device))

    def write_loss(self, epoch):
        for key, value in self.losses.items():
            if len(value) != 0:
                self.writer.add_scalar(key, value[-1], epoch)
