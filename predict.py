import pandas as pd   # add

@torch.no_grad()
def predict(self, data):
    self.eval()
    result_paras_np = []
    result_revert_prob_np = []
    result_cor_mat_np = []
    test_loader = self.load_dataloader(data, None, do_predict=True)

    max_save = 50      # 只存前 50 筆                           # add
    saved = 0          # add
    records = []       # 存放要輸出的資料                       # add

    for data in (pbar := tqdm(test_loader, ncols=100)):
        # 兼容 data 或 (data, labels)                           # add
        if isinstance(data, (tuple, list)) and len(data) == 2:  # add
            data, labels = data                                # add
        else:                                                  # add
            labels = None                                      # add

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

        # ===== 新增：存前 50 筆 Input (Norm_Spread) 與 μ/β + GT =====        # add
        x_np = data.cpu().detach().numpy()                                       # add
        mu_rtop, beta_rtop   = paras_np[:, 0], paras_np[:, 1]                    # add
        mu_top,  beta_top    = paras_np[:, 2], paras_np[:, 3]                    # add
        mu_close, beta_close = paras_np[:, 4], paras_np[:, 5]                    # add

        if labels is not None:                                                   # add
            y_np = labels.cpu().detach().numpy() if hasattr(labels, "cpu") else labels  # add
            gt_rtop, gt_top, gt_close = y_np[:, 1], y_np[:, 2], y_np[:, 3]       # add
        else:                                                                    # add
            gt_rtop = gt_top = gt_close = [None] * x_np.shape[0]                 # add

        for i in range(x_np.shape[0]):                                           # add
            if saved >= max_save:                                                # add
                break                                                            # add
            record = {                                                           # add
                "Norm_Spread": x_np[i, 0].tolist(),  # 存完整序列               # add
                "mu_rtop":   mu_rtop[i],                                         # add
                "beta_rtop": beta_rtop[i],                                       # add
                "mu_top":    mu_top[i],                                          # add
                "beta_top":  beta_top[i],                                        # add
                "mu_close":  mu_close[i],                                        # add
                "beta_close":beta_close[i],                                      # add
                "GT_Norm_Rtop":  gt_rtop[i],                                     # add
                "GT_Norm_Top":   gt_top[i],                                      # add
                "GT_Norm_Close": gt_close[i],                                    # add
            }                                                                    # add
            records.append(record)                                               # add
            saved += 1                                                           # add

        if saved >= max_save:                                                    # add
            break                                                                # add
        # =======================================================================

    # ===== 存成 CSV 檔 =====                                                   # add
    df = pd.DataFrame(records)                                                   # add
    df.to_csv("inference_results.csv", index=False)                              # add
    print("Saved first 50 inference results to inference_results.csv")           # add

    return np.array(result_revert_prob_np), np.array(result_paras_np), np.array(result_cor_mat_np)
