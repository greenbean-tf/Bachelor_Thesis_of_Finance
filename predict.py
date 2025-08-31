import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

@torch.no_grad()
def predict(self, data, labels=None):
    """
    推論 (inference)：
    - 讀取 data（以及可選的 labels），跑 forward
    - 蒐集所有 batch 的 revert_prob / paras / cor_mat
    - 另外把「前 50 筆」的 Norm_Spread、μ/β、以及 GT(Norm_Rtop/Top/Close) 輸出到 CSV
    """
    self.eval()
    result_paras_np = []
    result_revert_prob_np = []
    result_cor_mat_np = []

    # 注意：這裡把 labels 一起傳進去，Dataset 就會回傳 (x, y)
    test_loader = self.load_dataloader(data, labels, do_predict=True)

    max_save = 50      # 只輸出前 50 筆到 CSV
    saved = 0
    records = []       # 要寫入 CSV 的資料列

    # 保留原本的 for 行
    for data in (pbar := tqdm(test_loader, ncols=100)):
        # 兼容兩種 batch 格式：x 或 (x, y)
        if isinstance(data, (tuple, list)) and len(data) == 2:
            data, labels_batch = data
        else:
            labels_batch = None

        data = data.to(self.device)

        # forward：取得 (revert_prob, paras, cor_mat)
        revert_prob, paras, cor_mat = self.forward(data)

        # ====== 蒐集全部 batch 的輸出（原始行為） ======
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

        # ====== 只存前 50 筆到 CSV：Norm_Spread + μ/β + (可選)GT ======
        if saved < max_save:
            x_np = data.cpu().detach().numpy()               # (B, C=3, T)
            # paras 佈局：[μ_rtop, β_rtop, μ_top, β_top, μ_close, β_close]
            mu_rtop, beta_rtop   = paras_np[:, 0], paras_np[:, 1]
            mu_top,  beta_top    = paras_np[:, 2], paras_np[:, 3]
            mu_close, beta_close = paras_np[:, 4], paras_np[:, 5]

            if labels_batch is not None:
                y_np = labels_batch.cpu().detach().numpy() if hasattr(labels_batch, "cpu") else labels_batch
                gt_rtop, gt_top, gt_close = y_np[:, 1], y_np[:, 2], y_np[:, 3]
            else:
                # 若沒有 labels，填 NaN 以避免空白
                gt_rtop = np.full((x_np.shape[0],), np.nan, dtype=np.float32)
                gt_top  = np.full((x_np.shape[0],), np.nan, dtype=np.float32)
                gt_close= np.full((x_np.shape[0],), np.nan, dtype=np.float32)

            for i in range(x_np.shape[0]):
                if saved >= max_save:
                    break
                record = {
                    "Norm_Spread": x_np[i, 0].tolist(),  # 取第 0 通道（Norm_Spread）完整序列
                    "mu_rtop":   float(mu_rtop[i]),
                    "beta_rtop": float(beta_rtop[i]),
                    "mu_top":    float(mu_top[i]),
                    "beta_top":  float(beta_top[i]),
                    "mu_close":  float(mu_close[i]),
                    "beta_close":float(beta_close[i]),
                    "GT_Norm_Rtop":  float(gt_rtop[i]) if not np.isnan(gt_rtop[i]) else np.nan,
                    "GT_Norm_Top":   float(gt_top[i])  if not np.isnan(gt_top[i])  else np.nan,
                    "GT_Norm_Close": float(gt_close[i])if not np.isnan(gt_close[i])else np.nan,
                }
                records.append(record)
                saved += 1

        if saved >= max_save:
            break

    # ====== 寫出 CSV（前 50 筆樣本）======
    df = pd.DataFrame(records)
    df.to_csv("inference_results.csv", index=False)
    print("Saved first 50 inference results to inference_results.csv")

    # 回傳完整推論結果（全部樣本，而非只 50 筆）
    return (
        np.array(result_revert_prob_np),
        np.array(result_paras_np),
        np.array(result_cor_mat_np),
    )
