import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from tqdm.auto import tqdm
import os
from hyperparameters import *
import hyperparameters  # 額外引入模組本身，供 train_model 寫 .done 標記時讀取最新 log_path
from loss import GumIndLoss, NormIndLoss, GaussCopGumLoss, GaussCopNormLoss
from module import ResNetModule
from dataloader import eval_transform, train_transform
import dataloader


class GGPTS(nn.Module):
    def __init__(self, args):
        super(GGPTS, self).__init__()

        self.args = args
        self.best_model_name = None
        self.current_epoch = 0
        self.do_copula_model = args.do_copula_model
        self.set_loss(args.Loss)

        '''                       Model Architecture start                      '''
        self.resnet_module = ResNetModule()

        self.trans_enc_module = nn.TransformerEncoder(nn.TransformerEncoderLayer(1024, 4, 64, batch_first=True, dropout=0.5),8)

        feature_size = 128 * 40  #feature dimension * time dimension
        self.predicter1 = nn.Sequential(nn.Linear(feature_size, 64),
                                        nn.PReLU(),
                                        nn.Linear(64, 2),
                                        nn.LogSoftmax(dim=1))

        self.predicter2 = nn.Sequential(nn.Linear(feature_size, 64),
                                       nn.PReLU(),
                                       nn.Linear(64, 2))

        self.predicter3 = nn.Sequential(nn.Linear(feature_size, 64),
                                       nn.PReLU(),
                                       nn.Linear(64, 2))

        self.predicter4 = nn.Sequential(nn.Linear(feature_size, 64),
                                       nn.PReLU(),
                                       nn.Linear(64, 2))
        if self.do_copula_model:
            self.predicter5 = nn.Sequential(nn.Linear(feature_size, 64),
                                       nn.PReLU(),
                                       nn.Linear(64, 6),
                                       nn.Tanh())
        '''                       Model Architecture end                      '''

        self.early_stop_count = args.early_stop_count
        self.batch_size = args.batch_size
        self.device = args.device
        self.num_epochs = args.num_epochs
        self.num_workers = args.num_workers
        self.lr = args.lr
        self.norm_clip = args.norm_clip

        if args.optim == "Adam":
            self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        elif args.optim == "SGD":
            self.optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)
        elif args.optim == "Adadelta":
            self.optimizer = torch.optim.Adadelta(self.parameters(), lr=self.lr)
        elif args.optim == "Adagrad":
            self.optimizer = torch.optim.Adagrad(self.parameters(), lr=self.lr)
        elif args.optim == "RMSprop":
            self.optimizer = torch.optim.RMSprop(self.parameters(), lr=self.lr)
        else:
            raise ValueError("Wrong optimizer!")

        #self.scheduler = lr_scheduler.CyclicLR(self.optimizer, base_lr=self.lr, max_lr=self.lr*5, step_size_up=5000,
        #                                       mode='triangular2', cycle_momentum=False)
        self.scheduler = None

        self.save_root = args.save_root
        self.save_freq = args.save_freq
        os.makedirs(args.save_root, exist_ok=True)


    def forward(self, x):

        output = self.resnet_module(x)
        output = torch.permute(output, (0, 2, 1))

        output = self.trans_enc_module(output)
        output = output.view(-1, output.shape[1]*output.shape[2])

        r = self.predicter1(output)
        Rtop_distribution = self.predicter2(output)
        Top_distribution = self.predicter3(output)
        Close_distribution = self.predicter4(output)

        threshold_distributions = torch.cat((Rtop_distribution, Top_distribution, Close_distribution), dim=1)

        if self.do_copula_model:
            Sig = self.predicter5(output)
            correlation_matrix = Sig
            return r, threshold_distributions, correlation_matrix
        else:
            return r, threshold_distributions, None

    def train_model(self, train_data, train_labels, val_data, val_labels, do_test=False, test_data=None, test_labels=None,
                    resume_ckpt_path=None):
        if do_test:
            assert test_data is not None
            assert test_labels is not None

        # ── 接續訓練邏輯 ──────────────────────────────────────────────
        # 初始化訓練狀態（可能被 resume 覆寫）
        self._min_val_loss = 1000000
        self._min_val_count = 0
        self._experiment_folder = current_time  # 預設用本次啟動時間

        if resume_ckpt_path and os.path.exists(resume_ckpt_path):
            print(f"[Resume] 從 checkpoint 接續訓練：{resume_ckpt_path}")
            self.load_model(resume_ckpt_path)
            # 若 checkpoint 有記錄資料夾名稱，沿用同一個資料夾
            if self._experiment_folder is not None:
                print(f"[Resume] 沿用原實驗資料夾：{self._experiment_folder}")
            start_epoch = self.current_epoch + 1
            print(f"[Resume] 從 epoch {start_epoch} 繼續，目前最佳 val loss = {self._min_val_loss:.4f}")
        else:
            if resume_ckpt_path:
                print(f"[Resume] 找不到 checkpoint '{resume_ckpt_path}'，從頭開始訓練")
            self.history = {'train_loss': [], 'val_loss': [], 'test_loss': []}
            start_epoch = 0
        # ─────────────────────────────────────────────────────────────

        self.train()
        min_val_loss = self._min_val_loss
        min_val_count = self._min_val_count

        for epoch in range(start_epoch, self.num_epochs):
            epoch_train_loss = 0
            train_loader = self.load_dataloader(train_data, train_labels)
            self.current_epoch = epoch
            self.criterion.current_epoch = epoch # Updated by Colab Agent

            for (data, label) in (pbar := tqdm(train_loader, ncols=100)):
                data = data.to(self.device)
                label = label.to(self.device)

                outputs = self.forward(data)

                loss = self.criterion(outputs, label)

                epoch_train_loss += loss.detach().cpu() / len(train_loader)
                self.optimizer.zero_grad()

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), self.norm_clip)
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()
                    self.lr = self.scheduler.get_last_lr()[0]

                self.tqdm_bar('Train', pbar, loss.detach().cpu(), lr=self.lr)

            self.eval()
            epoch_val_loss = self.evaluate(val_data, val_labels, 'Validation')
            if do_test:
                epoch_test_loss = self.evaluate(test_data, test_labels, 'Test')
            self.train()

            print(f"Train set      loss : {epoch_train_loss.item():5f}")
            print(f"Validation set loss : {epoch_val_loss.item():5f}")
            if do_test:
                print(f"Test set loss       : {epoch_test_loss.item():5f}")

            self.history["train_loss"].append(epoch_train_loss.item())
            self.history["val_loss"].append(epoch_val_loss.item())
            if do_test:
                self.history["test_loss"].append(epoch_test_loss.item())

            # 使用 self._experiment_folder 確保 resume 後存到同一個資料夾
            exp_folder = self._experiment_folder
            if (epoch + 1) % self.save_freq == 0:
                if not os.path.exists(self.save_root + f"{exp_folder}/"):
                    os.makedirs(self.save_root + f"{exp_folder}/")
                self.save_model(self.save_root + f"{exp_folder}/Epoch={epoch+1}_ValLoss={epoch_val_loss.item():.4f}.pth")
                print(f'Epoch [{epoch + 1}/{self.num_epochs}], Loss: {epoch_train_loss.item():.4f}')

            if min_val_loss > epoch_val_loss:
                min_val_loss = epoch_val_loss
                min_val_count = 0
                self._min_val_loss = float(min_val_loss)
                self._min_val_count = min_val_count
                if self.best_model_name is not None and os.path.exists(self.best_model_name):
                    os.remove(self.best_model_name)
                self.best_model_name = (self.save_root +
                                   f"{exp_folder}/BestEpoch={epoch + 1}_ValLoss={epoch_val_loss.item():.4f}.pth")
                if not os.path.exists(self.save_root + f"{exp_folder}/"):
                    os.makedirs(self.save_root + f"{exp_folder}/")
                self.save_model(self.best_model_name)
            else:
                min_val_count += 1
                self._min_val_count = min_val_count
            if min_val_count >= self.early_stop_count:
                print("Early stop!")
                break

        # ── 訓練正常結束，寫入完成標記 ──────────────────────────────
        # 無論是 early stop 或跑完所有 epoch，都算正常結束
        # 中途崩潰不會執行到這裡，所以 .done 不會被建立
        try:
            done_path = hyperparameters.log_path.replace(".csv", ".done")
            with open(done_path, "w") as f:
                f.write(f"Training completed at epoch {self.current_epoch + 1}\n")
                f.write(f"Best val loss: {self._min_val_loss:.6f}\n")
                f.write(f"Best model: {self.best_model_name}\n")
            print(f"[Done] 訓練完成標記已寫入：{done_path}")
        except Exception as e:
            print(f"[Done] 警告：無法寫入完成標記：{e}")


    @torch.no_grad()
    def evaluate(self, val_data, val_labels, mode='Validation'):
        self.eval()
        epoch_val_loss = 0
        val_loader = self.load_dataloader(val_data, val_labels)

        for (data, label) in (pbar := tqdm(val_loader, ncols=100)):
            data = data.to(self.device)
            label = label.to(self.device)
            outputs = self.forward(data)
            loss = self.criterion(outputs, label)
            epoch_val_loss += loss.detach().cpu() / len(val_loader)
            self.tqdm_bar(mode, pbar, loss.detach().cpu(), lr=self.lr)
        return epoch_val_loss


    @torch.no_grad()
    def predict(self, data):
        self.eval()
        result_paras_np = []
        result_revert_prob_np = []
        result_cor_mat_np = []
        test_loader = self.load_dataloader(data, None, do_predict=True)

        for data in (pbar := tqdm(test_loader, ncols=100)):
            data = data.to(self.device)

            revert_prob, paras, cor_mat = self.forward(data)

            # concate each test batch into one output

            paras_np = paras.cpu().detach().numpy()
            for elem in paras_np:
                result_paras_np.append(elem)

            revert_prob_np = revert_prob.cpu().detach().numpy()
            for elem in revert_prob_np:
                result_revert_prob_np.append(elem)

            if cor_mat is not None:
                cor_mat_np = cor_mat.cpu().detach().numpy()
                for elem in cor_mat_np:
                    result_cor_mat_np.append(elem)

        return np.array(result_revert_prob_np), np.array(result_paras_np) , np.array(result_cor_mat_np)

    @torch.no_grad()
    def select_action(self, data):
        self.eval()

        revert_prob, paras, cor_mat = self.predict(data)
        paras = np.exp(paras)
        revert_prob = np.exp(revert_prob)

        if self.do_copula_model:
            return revert_prob, paras, cor_mat
        else:
            return revert_prob, paras


    def set_loss(self, loss):
        if loss == "GaussCopGumLoss":
            self.criterion = GaussCopGumLoss()
        elif loss == "GaussCopNormLoss":
            self.criterion = GaussCopNormLoss()
        elif loss == "GumIndLoss":
            self.criterion = GumIndLoss()
        elif loss == "NormIndLoss":
            self.criterion = NormIndLoss()
        else:
            raise ValueError("Loss only can be set as 'GaussCopGumLoss', 'GaussCopNormLoss', 'GumIndLoss', or 'NormIndLoss'.")


    def tqdm_bar(self, mode, pbar, loss, lr):
        pbar.set_description(f"({mode}) Epoch {self.current_epoch + 1}, lr:{lr:.8f}", refresh=False)
        pbar.set_postfix(loss=float(loss), refresh=False)
        pbar.refresh()


    def load_dataloader(self, data, labels, do_predict=False):

        transform = eval_transform if do_predict else train_transform

        dataset = dataloader.Dataset_PairsTrade(data=data,
                                                labels=labels,
                                                transform=transform)
        loader = DataLoader(dataset,
                            batch_size=self.batch_size,
                            num_workers=self.num_workers,
                            drop_last=False,
                            shuffle=(False if do_predict else True))
        return loader


    def plot_train_history(self,data_name='',experiment_file=None, show=False):

        if not experiment_file:
            experiment_file = current_time

        plt.plot(range(len(self.history["train_loss"])), self.history["train_loss"], label="Train")
        plt.plot(range(len(self.history["val_loss"])), self.history["val_loss"], label="Valid")
        plt.plot(range(len(self.history["test_loss"])), self.history["test_loss"], label="Test")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss/Epoch")
        plt.legend()

        if show:
            plt.show()
        else:
            if not os.path.exists(PathRoot + f"Record/{experiment_file}/"):
                os.makedirs(PathRoot + f"Record/{experiment_file}/")
            plt.savefig(PathRoot + f"Record/{experiment_file}/" +
                    f"{data_name}_Training_loss.png")
        plt.close()


    def load_model(self, ckpt_path):
        if ckpt_path != None:
            checkpoint = torch.load(ckpt_path, map_location=self.device)
            self.load_state_dict(checkpoint['state_dict'], strict=True)
            self.lr = checkpoint['lr']
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.current_epoch = checkpoint['last_epoch']
            self.history = checkpoint['history']
            self._min_val_loss = checkpoint.get('min_val_loss', 1000000)
            self._min_val_count = checkpoint.get('min_val_count', 0)
            self._experiment_folder = checkpoint.get('experiment_folder', None)
            self.best_model_name = checkpoint.get('best_model_name', None)
            # 舊 checkpoint 沒有存 best_model_name，自動從同資料夾尋找
            if self.best_model_name is None or not os.path.exists(self.best_model_name):
                ckpt_dir = os.path.dirname(ckpt_path)
                candidates = [f for f in os.listdir(ckpt_dir) if f.startswith("BestEpoch=")]
                if candidates:
                    # 有多個的話取 ValLoss 最低的
                    def parse_val_loss(fname):
                        try: return float(fname.split("ValLoss=")[1].replace(".pth", ""))
                        except: return float("inf")
                    best_file = min(candidates, key=parse_val_loss)
                    self.best_model_name = os.path.join(ckpt_dir, best_file)
                    print(f"[Load] best model 自動找到：{self.best_model_name}")
                else:
                    print(f"[Load] 警告：找不到 BestEpoch checkpoint，best_model_name = None")
        else:
            raise ValueError("ckpt_path is None!")


    def save_model(self, ckpt_path):
        torch.save({
            "state_dict": self.state_dict(),
            "optimizer": self.optimizer.state_dict(),   # 修正：存 optimizer 狀態而非 model weights
            "lr": self.lr,
            "last_epoch": self.current_epoch,
            "history": self.history,
            "min_val_loss": self._min_val_loss,         # 新增：保存最佳 val loss 供 resume 使用
            "min_val_count": self._min_val_count,       # 新增：保存 early stop 計數器
            "experiment_folder": self._experiment_folder,  # 新增：保存資料夾名稱，resume 時沿用同一個
            "best_model_name": self.best_model_name,
        }, ckpt_path)
        print(f"save ckpt to {ckpt_path}")

