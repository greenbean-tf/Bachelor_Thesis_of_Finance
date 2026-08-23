import torch
import torch.nn as nn
from tqdm.auto import tqdm
import numpy as np
import utils
from MyTool import append_debug_log
import hyperparameters

class GumIndLoss(nn.Module):
    def forward(self, y_pred, y_true):
        Revert = torch.unsqueeze(y_true[:, 0], 1).long()
        Rtop = torch.unsqueeze(y_true[:, 1],1)
        Top = torch.unsqueeze(y_true[:, 2], 1)
        Close = torch.unsqueeze(y_true[:, 3], 1)

        r = y_pred[0]
        rtop = torch.unsqueeze(torch.exp(y_pred[1][:, 0]),1)
        sigma_rtop = torch.unsqueeze(torch.exp(y_pred[1][:, 1]),1)
        top = torch.unsqueeze(torch.exp(y_pred[1][:, 2]), 1)
        sigma_top = torch.unsqueeze(torch.exp(y_pred[1][:, 3]), 1)
        close = torch.unsqueeze(torch.exp(y_pred[1][:, 4]), 1)
        sigma_close = torch.unsqueeze(torch.exp(y_pred[1][:, 5]), 1)

        eps = torch.tensor(1e-4)
        rtop_z = torch.clamp((rtop-Rtop)/torch.max(sigma_rtop,eps),-10,10)
        Rtop_Gumbel_loss = ((torch.exp(rtop_z)
                                - rtop_z
                                + torch.log(torch.max(sigma_rtop,eps))) )
        top_z = (top - Top) / torch.max(sigma_top, eps)
        Top_Gumbel_loss = ((torch.exp(top_z)
                                       - top_z
                                       + torch.log(torch.max(sigma_top, eps))) )
        Close_Normal_loss = 1/2 * (torch.square((close - Close) / torch.max(sigma_close, eps))
                                     + torch.log(torch.square(torch.max(sigma_close, eps))))

        prob_loss = Revert * (Rtop_Gumbel_loss + Top_Gumbel_loss + Close_Normal_loss)

        return (nn.CrossEntropyLoss()(r, torch.squeeze(Revert))
                + torch.mean(prob_loss))

class NormIndLoss(nn.Module):
    def forward(self, y_pred, y_true):
        Revert = torch.unsqueeze(y_true[:, 0], 1).long()
        Rtop = torch.unsqueeze(y_true[:, 1],1)
        Top = torch.unsqueeze(y_true[:, 2], 1)
        Close = torch.unsqueeze(y_true[:, 3], 1)

        r = y_pred[0]
        rtop = torch.unsqueeze(torch.exp(y_pred[1][:, 0]),1)
        sigma_rtop = torch.unsqueeze(torch.exp(y_pred[1][:, 1]),1)
        top = torch.unsqueeze(torch.exp(y_pred[1][:, 2]), 1)
        sigma_top = torch.unsqueeze(torch.exp(y_pred[1][:, 3]), 1)
        close = torch.unsqueeze(torch.exp(y_pred[1][:, 4]), 1)
        sigma_close = torch.unsqueeze(torch.exp(y_pred[1][:, 5]), 1)

        eps = torch.tensor(1e-4)

        Rtop_Normal_loss = 1/2 * (torch.square((rtop - Rtop) / torch.max(sigma_rtop, eps))
                                       + torch.log(torch.square(torch.max(sigma_rtop, eps))))
        Top_Normal_loss = 1/2 * (torch.square((top - Top) / torch.max(sigma_top, eps))
                                       + torch.log(torch.square(torch.max(sigma_top, eps))))
        Close_Normal_loss = 1/2 * (torch.square((close - Close) / torch.max(sigma_close, eps))
                                     + torch.log(torch.square(torch.max(sigma_close, eps))))

        prob_loss = Revert*(Rtop_Normal_loss + Top_Normal_loss + Close_Normal_loss)

        return (nn.CrossEntropyLoss()(r, torch.squeeze(Revert))
                + torch.mean(prob_loss))

class GaussCopGumLoss(nn.Module):
    def forward(self, y_pred, y_true_with_id):
        # 1. 把 Data_ID 剝離出來
        data_id = y_true_with_id[:, 0].long()
        # 2. 剩下的才是原本的 labels
        y_true = y_true_with_id[:, 1:]

        Revert = torch.unsqueeze(y_true[:, 0], 1).long()
        Rtop = torch.unsqueeze(y_true[:, 1],1)
        Top = torch.unsqueeze(y_true[:, 2], 1)
        Close = torch.unsqueeze(y_true[:, 3], 1)

        r = y_pred[0]
        rtop = torch.unsqueeze(torch.exp(y_pred[1][:, 0]),1)
        sigma_rtop = torch.unsqueeze(torch.exp(y_pred[1][:, 1]),1)
        top = torch.unsqueeze(torch.exp(y_pred[1][:, 2]), 1)
        sigma_top = torch.unsqueeze(torch.exp(y_pred[1][:, 3]), 1)
        close = torch.unsqueeze(torch.exp(y_pred[1][:, 4]), 1)
        sigma_close = torch.unsqueeze(torch.exp(y_pred[1][:, 5]), 1)

        r12 = y_pred[2][:, 0]
        r13 = y_pred[2][:, 1]
        r23 = y_pred[2][:, 2]

        R = torch.eye(3).unsqueeze(0).expand(len(y_true), -1, -1).to('cuda')

        R[:, 0, 1] = r12
        R[:, 1, 0] = r12
        R[:, 0, 2] = r13
        R[:, 2, 0] = r13
        R[:, 1, 2] = r23
        R[:, 2, 1] = r23

        
        
        eps = torch.tensor(1e-4).to('cuda')

        #ill-conditioned?（相關係數矩陣接近奇異，導致 exp_term 的矩陣求逆不穩定）
        eigvals = torch.linalg.eigvalsh(R)
        lambda_max = eigvals.max(dim=-1).values
        lambda_min = eigvals.min(dim=-1).values
        lambda_min = torch.clamp(lambda_min, min=1e-12)
        cond = lambda_max / lambda_min

        # 設定條件數閥值 (下修到 1e7)
        cond_threshold = 1e7
        cond_bad_mask = cond > cond_threshold

        # sigma 崩塌？(rtop/top/close 的離散度參數被壓到接近 0)
        # 這是造成 loss 跑到異常負值的第二條獨立路徑，跟相關係數矩陣是否
        # 接近奇異完全無關——即使 R 條件數正常，sigma 崩塌一樣會讓
        # log(sigma^2) 項爆出很大的負值。原本只監控 R 的條件數，會漏掉
        # 這條路徑造成的異常樣本，永遠不會被記錄進清洗名單。
        sigma_threshold = hyperparameters.sigma_collapse_threshold
        sigma_flat = torch.cat([sigma_rtop, sigma_top, sigma_close], dim=1)
        sigma_bad_mask = (sigma_flat < sigma_threshold).any(dim=1)

        bad_mask = cond_bad_mask | sigma_bad_mask
        good_mask = ~bad_mask

        # 如果有壞資料，記錄它們的 ID（依照觸發原因分別記錄，方便之後分析）
        if cond_bad_mask.any():
            bad_indices = cond_bad_mask.nonzero(as_tuple=True)[0]
            for idx in bad_indices:
                idx = idx.item()
                real_data_id = data_id[idx].item()

                append_debug_log(hyperparameters.log_path, {
                    "Data_ID": real_data_id,
                    "epoch": getattr(self, "current_epoch", None),
                    "cond_number": cond[idx].item(),
                    "r12": r12[idx].item(),
                    "r13": r13[idx].item(),
                    "r23": r23[idx].item()
                })

        if sigma_bad_mask.any():
            bad_indices = sigma_bad_mask.nonzero(as_tuple=True)[0]
            for idx in bad_indices:
                idx = idx.item()
                real_data_id = data_id[idx].item()

                append_debug_log(hyperparameters.sigma_log_path, {
                    "Data_ID": real_data_id,
                    "epoch": getattr(self, "current_epoch", None),
                    "sigma_rtop": sigma_rtop[idx].item(),
                    "sigma_top": sigma_top[idx].item(),
                    "sigma_close": sigma_close[idx].item()
                })

        # 如果整個 Batch 全部都是壞資料，才丟出常數 Loss
        if not good_mask.any():
            # 💡 這裡加 (r.sum() * 0) 是一個 PyTorch 技巧：
            # 為了避免 return 純常數導致「計算圖斷裂(grad detached)」而引發 backpropagation 報錯，
            # 加上 r.sum()*0 可以確保它連在計算圖上，但梯度為 0，安全跳過這一步。
            return (r.sum() * 0.0) + 500.0

        # === 🌟 3. 套用遮罩，只留下正常資料！ ===
        # 把所有的變數都只保留 good_mask 的部分
        R = R[good_mask]
        Revert = Revert[good_mask]
        Rtop = Rtop[good_mask]
        Top = Top[good_mask]
        Close = Close[good_mask]
        r = r[good_mask]
        rtop = rtop[good_mask]
        sigma_rtop = sigma_rtop[good_mask]
        top = top[good_mask]
        sigma_top = sigma_top[good_mask]
        close = close[good_mask]
        sigma_close = sigma_close[good_mask]
        # 連 data_id 也要跟著縮小，這樣後面的 Copula_loss 記錄才會對得起來
        data_id = data_id[good_mask]

        detR = torch.unsqueeze(torch.det(R),1)

        q1 = utils.GumbelCDF(Rtop, u=rtop, s=sigma_rtop, eps=eps, do_torch=True)
        q2 = utils.GumbelCDF(Top, u=top, s=sigma_top, eps=eps, do_torch=True)
        q3 = utils.NormalCDF(Close, u=close, s=sigma_close, eps=eps, do_torch=True)

        q = torch.concatenate((q1, q2, q3), dim=1)

        quantile_vector = utils.Normal_distribution_q(0, 1, q, True).unsqueeze(1)
        quantile_vector = torch.clamp(quantile_vector, -3, 3)
        quantile_vector_T = torch.transpose(quantile_vector, 1, 2)
        exp_term = 1/2 * (
            torch.matmul(torch.matmul(quantile_vector, ((torch.linalg.inv(R) - torch.eye(3).to('cuda')))),
                         quantile_vector_T))
        exp_term = torch.squeeze(exp_term,dim=1)

        rtop_z = (rtop - Rtop) / torch.max(sigma_rtop, eps)
        Rtop_Gumbel_loss = (torch.exp(rtop_z)
                                      - rtop_z
                                      + torch.log(torch.max(sigma_rtop, eps)))
        top_z = (top - Top) / torch.max(sigma_top, eps)
        Top_Gumbel_loss = (torch.exp(top_z)
                                     - top_z
                                     + torch.log(torch.max(sigma_top, eps)))
        Close_Normal_loss = 1/2 * (torch.square((close - Close) / torch.max(sigma_close, eps))
                                       + torch.log(torch.square(torch.max(sigma_close, eps))))

        cGaussian = 1/2*torch.log(torch.max(detR,eps)) + exp_term

        Copula_loss = Revert*(cGaussian + Rtop_Gumbel_loss + Top_Gumbel_loss + Close_Normal_loss)

        """
        # 找出 batch 中所有 Copula_loss > 0 的樣本
        good_idx = (Copula_loss > 0).nonzero(as_tuple=True)[0]

        if len(good_idx) > 0:
            for idx in good_idx:
                idx = idx.item()
                
                print("=== Abnormal sample detected ===")
                print("Sample index:", idx)
                print("Copula_loss:", Copula_loss[idx].item())
                
                print("\n--- Copula exp term ---")
                print("exp_term:", exp_term[idx].item())


                print("\n--- Marginal Loss Terms ---")
                print("Rtop_Gumbel_loss:", Rtop_Gumbel_loss[idx].item())
                print("Top_Gumbel_loss :", Top_Gumbel_loss[idx].item())
                print("Close_Normal_loss:", Close_Normal_loss[idx].item())


                # labels
                print("\n--- Labels ---")
                print("Revert:", Revert[idx].item())
                print("Rtop:", Rtop[idx].item())
                print("Top:", Top[idx].item())
                print("Close:", Close[idx].item())

                # predicted μ
                print("\n--- Predicted μ ---")
                print("μ rtop:", rtop[idx].item())
                print("μ top :", top[idx].item())
                print("μ close:", close[idx].item())

                # predicted σ
                print("\n--- Predicted σ ---")
                print("σ rtop:", sigma_rtop[idx].item())
                print("σ top :", sigma_top[idx].item())
                print("σ close:", sigma_close[idx].item())

                # correlation
                print("\n--- Correlation ---")
                print("ρ12:", r12[idx].item())
                print("ρ13:", r13[idx].item())
                print("ρ23:", r23[idx].item())

                # determinant
                print("\n--- det(R) ---")
                print("det(R):", detR[idx].item())
                
                # Write to CSV
                log_path = "./output/lossmorethan0_compare.csv"
                append_debug_log(log_path, {
                    "sample_idx": idx,
                    "copula_loss": Copula_loss[idx].item(),
                    "exp_term": exp_term[idx].item(),

                    "label_revert": Revert[idx].item(),
                    "label_rtop": Rtop[idx].item(),
                    "label_top": Top[idx].item(),
                    "label_close": Close[idx].item(),

                    "mu_rtop": rtop[idx].item(),
                    "mu_top": top[idx].item(),
                    "mu_close": close[idx].item(),

                    "sigma_rtop": sigma_rtop[idx].item(),
                    "sigma_top": sigma_top[idx].item(),
                    "sigma_close": sigma_close[idx].item(),

                    "rho12": r12[idx].item(),
                    "rho13": r13[idx].item(),
                    "rho23": r23[idx].item(),

                    "detR": detR[idx].item(),
                    "Rtop_Gumbel": Rtop_Gumbel_loss[idx].item(),
                    "Top_Gumbel": Top_Gumbel_loss[idx].item(),
                    "Close_Normal": Close_Normal_loss[idx].item(),
                })


        
        # 找出 batch 中所有 Copula_loss < 0 的樣本
        bad_idx = (Copula_loss < 0).nonzero(as_tuple=True)[0]

        if len(bad_idx) > 0:
            for idx in bad_idx:
                idx = idx.item()
                
                print("=== Abnormal sample detected ===")
                print("Sample index:", idx)
                print("Copula_loss:", Copula_loss[idx].item())
                
                print("\n--- Copula exp term ---")
                print("exp_term:", exp_term[idx].item())


                print("\n--- Marginal Loss Terms ---")
                print("Rtop_Gumbel_loss:", Rtop_Gumbel_loss[idx].item())
                print("Top_Gumbel_loss :", Top_Gumbel_loss[idx].item())
                print("Close_Normal_loss:", Close_Normal_loss[idx].item())


                # labels
                print("\n--- Labels ---")
                print("Revert:", Revert[idx].item())
                print("Rtop:", Rtop[idx].item())
                print("Top:", Top[idx].item())
                print("Close:", Close[idx].item())

                # predicted μ
                print("\n--- Predicted μ ---")
                print("μ rtop:", rtop[idx].item())
                print("μ top :", top[idx].item())
                print("μ close:", close[idx].item())

                # predicted σ
                print("\n--- Predicted σ ---")
                print("σ rtop:", sigma_rtop[idx].item())
                print("σ top :", sigma_top[idx].item())
                print("σ close:", sigma_close[idx].item())

                # correlation
                print("\n--- Correlation ---")
                print("ρ12:", r12[idx].item())
                print("ρ13:", r13[idx].item())
                print("ρ23:", r23[idx].item())

                # determinant
                print("\n--- det(R) ---")
                print("det(R):", detR[idx].item())
                
                # Write to CSV
                log_path = "./output/losslessthan0_compare.csv"
                append_debug_log(log_path, {
                    "sample_idx": idx,
                    "copula_loss": Copula_loss[idx].item(),
                    "exp_term": exp_term[idx].item(),

                    "label_revert": Revert[idx].item(),
                    "label_rtop": Rtop[idx].item(),
                    "label_top": Top[idx].item(),
                    "label_close": Close[idx].item(),

                    "mu_rtop": rtop[idx].item(),
                    "mu_top": top[idx].item(),
                    "mu_close": close[idx].item(),

                    "sigma_rtop": sigma_rtop[idx].item(),
                    "sigma_top": sigma_top[idx].item(),
                    "sigma_close": sigma_close[idx].item(),

                    "rho12": r12[idx].item(),
                    "rho13": r13[idx].item(),
                    "rho23": r23[idx].item(),

                    "detR": detR[idx].item(),
                    "Rtop_Gumbel": Rtop_Gumbel_loss[idx].item(),
                    "Top_Gumbel": Top_Gumbel_loss[idx].item(),
                    "Close_Normal": Close_Normal_loss[idx].item(),
                })

        """

      
        
        return nn.CrossEntropyLoss()(r, torch.squeeze(Revert, dim=1)) + torch.mean(Copula_loss)

class GaussCopNormLoss(nn.Module):
    def forward(self, y_pred, y_true):
        Revert = torch.unsqueeze(y_true[:, 0], 1).long()
        Rtop = torch.unsqueeze(y_true[:, 1],1)
        Top = torch.unsqueeze(y_true[:, 2], 1)
        Close = torch.unsqueeze(y_true[:, 3], 1)

        r = y_pred[0]
        rtop = torch.unsqueeze(torch.exp(y_pred[1][:, 0]),1)
        sigma_rtop = torch.unsqueeze(torch.exp(y_pred[1][:, 1]),1)
        top = torch.unsqueeze(torch.exp(y_pred[1][:, 2]), 1)
        sigma_top = torch.unsqueeze(torch.exp(y_pred[1][:, 3]), 1)
        close = torch.unsqueeze(torch.exp(y_pred[1][:, 4]), 1)
        sigma_close = torch.unsqueeze(torch.exp(y_pred[1][:, 5]), 1)

        r12 = y_pred[2][:, 0]
        r13 = y_pred[2][:, 1]
        r23 = y_pred[2][:, 2]

        R = torch.eye(3).unsqueeze(0).expand(len(y_true), -1, -1).to('cuda')

        R[:, 0, 1] = r12
        R[:, 1, 0] = r12
        R[:, 0, 2] = r13
        R[:, 2, 0] = r13
        R[:, 1, 2] = r23
        R[:, 2, 1] = r23

        eps = torch.tensor(1e-4).to('cuda')

        detR = torch.unsqueeze(torch.det(R), 1)

        q1 = utils.NormalCDF(Rtop, u=rtop, s=sigma_rtop, eps=eps, do_torch=True)
        q2 = utils.NormalCDF(Top, u=top, s=sigma_top, eps=eps, do_torch=True)
        q3 = utils.NormalCDF(Close, u=close, s=sigma_close, eps=eps, do_torch=True)

        q = torch.concatenate((q1, q2, q3), dim=1)

        quantile_vector = utils.Normal_distribution_q(0, 1, q, True).unsqueeze(1)
        quantile_vector = torch.clamp(quantile_vector, -3, 3)
        quantile_vector_T = torch.transpose(quantile_vector, 1, 2)
        exp_term = 1 / 2 * (
            torch.matmul(torch.matmul(quantile_vector, ((torch.linalg.inv(R) - torch.eye(3).to('cuda')))),
                         quantile_vector_T))
        exp_term = torch.squeeze(exp_term, dim=1)

        Rtop_Normal_loss = 1/2 * (torch.square((rtop - Rtop) / torch.max(sigma_rtop, eps))
                                             + torch.log(torch.square(torch.max(sigma_rtop, eps))))
        Top_Normal_loss = 1/2 *  (torch.square((top - Top) / torch.max(sigma_top, eps))
                                            + torch.log(torch.square(torch.max(sigma_top, eps))))
        Close_Normal_loss = 1/2 * (torch.square((close - Close) / torch.max(sigma_close, eps))
                                     + torch.log(torch.square(torch.max(sigma_close, eps))))

        cGaussian = 1/2 * torch.log(torch.max(detR, eps)) + exp_term

        Copula_loss = Revert * (cGaussian + Rtop_Normal_loss + Top_Normal_loss + Close_Normal_loss)

        return nn.CrossEntropyLoss()(r, torch.squeeze(Revert)) + torch.mean(Copula_loss)