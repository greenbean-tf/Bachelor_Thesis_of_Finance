import pandas as pd
import numpy as np
import preprocess
from backtest import Backtest
import model
import os
import utils
import hyperparameters
from hyperparameters import args, split_info, PathRoot, preprocess_data_path
import torch
from tqdm.auto import tqdm
torch.autograd.set_detect_anomaly(True)

def main(args, split_info, mode="train", model_path=None, main_time=None):
    # Store period split infomation
    split_dict = split_info.split_dict
    start_year = split_info.start_year
    start_month = split_info.start_month
    end_year = split_info.end_year
    end_month = split_info.end_month

    experiment_file = f"{main_time}/"

    # create preprocess data(.pickle) and read pickle
    # the reason why use pickle rather than csv is that csv turn list into string and it is annoying
    '''if not os.path.exists(preprocess_data_path):
        preprocess.create_preprocess_data(start_year=start_year,
                                          start_month=start_month,
                                          end_year=end_year,
                                          end_month=end_year,
                                          save_path=preprocess_data_path)'''
    '''
    preprocess_data1 = pd.read_pickle(PathRoot+f'preprocess_data_{2018}_{10}_to_{2019}_{9}_0.pickle')
    preprocess_data2 = pd.read_pickle(PathRoot+f'preprocess_data_{2019}_{10}_to_{2020}_{9}_0.pickle')
    preprocess_data3 = pd.read_pickle(PathRoot+f'preprocess_data_{2020}_{10}_to_{2021}_{9}_0.pickle')
    preprocess_data4 = pd.read_pickle(PathRoot+f'preprocess_data_{2021}_{10}_to_{2022}_{9}_0.pickle')
    preprocess_data4 = preprocess_data4[preprocess_data4['Date']<20220101]
    #preprocess_data5 = pd.read_pickle(PathRoot+"preprocess_data/"+f'preprocess_data_{2022}_{10}_to_{2023}_{10}.pickle')
    preprocess_data = pd.concat([preprocess_data1, preprocess_data2, preprocess_data3, preprocess_data4])
    #preprocess_data = pd.concat([preprocess_data4,preprocess_data5])
    #preprocess_data = pd.read_pickle(preprocess_data_path)
    '''

    '''
    # glabal learning
    preprocess_data1_0 = pd.read_pickle(PathRoot+f'preprocess_data_{2018}_{10}_to_{2019}_{9}_0.pickle')
    preprocess_data1_1 = pd.read_pickle(PathRoot+f'preprocess_data_{2018}_{10}_to_{2019}_{9}_1.pickle')
    preprocess_data1_2 = pd.read_pickle(PathRoot+f'preprocess_data_{2018}_{10}_to_{2019}_{9}_2.pickle')
    preprocess_data2_0 = pd.read_pickle(PathRoot+f'preprocess_data_{2019}_{10}_to_{2020}_{9}_0.pickle')
    preprocess_data2_1 = pd.read_pickle(PathRoot+f'preprocess_data_{2019}_{10}_to_{2020}_{9}_1.pickle')
    preprocess_data2_2 = pd.read_pickle(PathRoot+f'preprocess_data_{2019}_{10}_to_{2020}_{9}_2.pickle')
    preprocess_data3_0 = pd.read_pickle(PathRoot+f'preprocess_data_{2020}_{10}_to_{2021}_{9}_0.pickle')
    preprocess_data3_1 = pd.read_pickle(PathRoot+f'preprocess_data_{2020}_{10}_to_{2021}_{9}_1.pickle')
    preprocess_data3_2 = pd.read_pickle(PathRoot+f'preprocess_data_{2020}_{10}_to_{2021}_{9}_2.pickle')
    preprocess_data4_0 = pd.read_pickle(PathRoot+f'preprocess_data_{2021}_{10}_to_{2022}_{9}_0.pickle')
    preprocess_data4_1 = pd.read_pickle(PathRoot+f'preprocess_data_{2021}_{10}_to_{2022}_{9}_1.pickle')
    preprocess_data4_2 = pd.read_pickle(PathRoot+f'preprocess_data_{2021}_{10}_to_{2022}_{9}_2.pickle')
    preprocess_data5_0 = pd.read_pickle(PathRoot+f'preprocess_data_{2022}_{10}_to_{2023}_{10}_0.pickle')
    preprocess_data5_1 = pd.read_pickle(PathRoot+f'preprocess_data_{2022}_{10}_to_{2023}_{10}_1.pickle')
    preprocess_data5_2 = pd.read_pickle(PathRoot+f'preprocess_data_{2022}_{10}_to_{2023}_{10}_2.pickle')
    preprocess_data = pd.concat([preprocess_data1_0, preprocess_data1_1, preprocess_data1_2, 
                                preprocess_data2_0, preprocess_data2_1, preprocess_data2_2, 
                                preprocess_data3_0, preprocess_data3_1, preprocess_data3_2,  
                                preprocess_data4_0, preprocess_data4_1, preprocess_data4_2,
                                preprocess_data5_0, preprocess_data5_1, preprocess_data5_2
                                ])
    #preprocess_data = pd.concat([preprocess_data4,preprocess_data5])
    #preprocess_data = pd.read_pickle(preprocess_data_path)
    '''
    inference_dir = PathRoot + "inference_cache/"

    if mode == "backtest_from_saved" and os.path.exists(inference_dir + "test_df.pkl"):
        # 方案 A：直接載入 test_df.pkl，跳過 10-30 GB 的 cleaned_data pickle
        print("backtest_from_saved: 直接載入 test_df.pkl，跳過全量 preprocess")
        test = pd.read_pickle(inference_dir + "test_df.pkl")
        train = val = preprocess_data = None
        print(f"  test DataFrame 載入完成，共 {len(test):,} 筆")
    else:
        # 改成直接讀取清洗好的乾淨資料
        print("載入清洗後的乾淨資料...")
        if hyperparameters.data_version_override is not None:
            n = int(hyperparameters.data_version_override)
            print(f"[Data Version Override] 強制使用 v{n}（忽略自動偵測最新版本的邏輯）")
        else:
            n = 0
            while os.path.exists(f"{'/content/GGWP_US_LLTL/data_cleaning/cleaned_data_v'}{n+1}.pickle"):
                n += 1
        latest_path = f"{'/content/GGWP_US_LLTL/data_cleaning/cleaned_data_v'}{n}.pickle"
        print(f"讀取版本：v{n}，路徑：{latest_path}")
        preprocess_data = pd.read_pickle(latest_path)

        preprocess_data.reset_index(drop=True, inplace=True)
        preprocess_data['Data_ID'] = preprocess_data.index

        utils.check_nan(preprocess_data)

        train, val, test = preprocess.Split_data(preprocess_data,
                                                 by_time=True,
                                                 split_dates=split_dict)

    if mode == "train":
        # 訓練模式：需要把 DataFrame 轉成 numpy array 才能餵給 DataLoader
        # 三個 split 合計約 15~25 GB RAM，是正常必要開銷
        train_data = np.stack([np.stack(train['Norm_Spread'].values),
                               np.stack(train['S1_Return'].values),
                               np.stack(train['S2_Return'].values)], axis=1)
        train_labels = train[['Data_ID','Revert', 'Norm_Rtop', 'Norm_Top', 'Norm_Close', 'Norm_Tax']].values

        val_data = np.stack([np.stack(val['Norm_Spread'].values),
                             np.stack(val['S1_Return'].values),
                             np.stack(val['S2_Return'].values)], axis=1)
        val_labels = val[['Data_ID','Revert', 'Norm_Rtop', 'Norm_Top', 'Norm_Close', 'Norm_Tax']].values

        test_data = np.stack([np.stack(test['Norm_Spread'].values),
                             np.stack(test['S1_Return'].values),
                             np.stack(test['S2_Return'].values)], axis=1)
        test_labels = test[['Data_ID','Revert', 'Norm_Rtop', 'Norm_Top', 'Norm_Close', 'Norm_Tax']].values

        print("Train data shape:", train_data.shape)
        print("Valid data shape:", val_data.shape)
        print("Test  data shape:", test_data.shape)
    elif mode == "inference_only":
        # inference_only 模式：只需要 test_data 餵給模型，其餘不需要
        test_data = np.stack([np.stack(test['Norm_Spread'].values),
                              np.stack(test['S1_Return'].values),
                              np.stack(test['S2_Return'].values)], axis=1)
        train_data = val_data = None
        train_labels = val_labels = test_labels = None
        print(f"Inference mode: test_data built  |  test 資料筆數 = {len(test):,}")
    else:
        # test / backtest_from_saved / baseline：backtest 直接從 test DataFrame 讀取，
        # 完全不需要 numpy array。跳過轉換可節省 15~25 GB RAM 和數分鐘的處理時間。
        train_data = val_data = test_data = None
        train_labels = val_labels = test_labels = None
        print(f"跳過 numpy 轉換  |  test 資料筆數 = {len(test):,}")

  #########################################################

    if mode == "inference_only":
        # ── GPU Session：跑 model inference，結果存到 inference_cache/ ──────────
        PT_model = model.GGPTS(args).to(args.device)
        if model_path:
            print(f"Pretrained model loaded: {model_path}")
            PT_model.load_model(model_path)
            # model_path 通常是 find_latest_checkpoint() 找到的「最後一個檔案」（依修改時間），
            # 也就是 early stop 前的最後一個 epoch，不是驗證集表現最好的那個。
            # load_model() 內部會從 checkpoint 或同資料夾自動找出 best_model_name，
            # 這裡必須再載入一次才能真正套用最佳權重（跟 mode=="test" 的邏輯保持一致）。
            if PT_model.best_model_name and os.path.exists(PT_model.best_model_name):
                print(f"[Inference] 改用驗證集最佳 checkpoint：{PT_model.best_model_name}")
                PT_model.load_model(PT_model.best_model_name)
            else:
                print("[Inference] 警告：找不到 best_model_name，沿用 model_path 的權重")
        elif PT_model.best_model_name and os.path.exists(PT_model.best_model_name):
            PT_model.load_model(PT_model.best_model_name)
        else:
            print("[Inference] No checkpoint found, using random weights")

        print("Running model inference on GPU...")
        if args.do_copula_model:
            revert_prob, paras, cor_mat = PT_model.select_action(test_data)
        else:
            revert_prob, paras = PT_model.select_action(test_data)
            cor_mat = np.zeros((len(test_data), 3), dtype=np.float32)

        os.makedirs(inference_dir, exist_ok=True)
        np.save(inference_dir + "paras.npy",       paras)
        np.save(inference_dir + "revert_prob.npy", revert_prob)
        np.save(inference_dir + "cor_mat.npy",     cor_mat)
        test.to_pickle(inference_dir + "test_df.pkl")
        print(f"Inference cache saved → {inference_dir}")
        print(f"  paras:       {paras.shape}  ({paras.nbytes/1e6:.1f} MB)")
        print(f"  revert_prob: {revert_prob.shape}  ({revert_prob.nbytes/1e6:.1f} MB)")
        print(f"  cor_mat:     {cor_mat.shape}  ({cor_mat.nbytes/1e6:.1f} MB)")
        print(f"  test_df:     {len(test):,} rows  → test_df.pkl")

    elif mode == "backtest_from_saved":
        # ── CPU Session：從 inference_cache/ 載入，純 CPU 跑回測 ─────────────
        paras       = np.load(inference_dir + "paras.npy")
        revert_prob = np.load(inference_dir + "revert_prob.npy")
        cor_mat     = np.load(inference_dir + "cor_mat.npy")
        print(f"Loaded inference cache from {inference_dir}")
        print(f"  paras:       {paras.shape}")
        print(f"  revert_prob: {revert_prob.shape}")
        print(f"  cor_mat:     {cor_mat.shape}")

        # ── post-hoc sigma 校正（驗證σ低估假說，不重跑模型推論）──────────
        # paras 欄位順序：[rtop, sigma_rtop, top, sigma_top, close, sigma_close]
        data_name_suffix = ""
        if hyperparameters.do_posthoc_sigma_correction:
            f = hyperparameters.posthoc_sigma_correction_factors
            print(f"[Post-hoc Sigma Correction] 套用校正倍率：rtop x{f['rtop']}, top x{f['top']}, close x{f['close']}")
            print(f"  校正前 mean sigma → rtop={paras[:,1].mean():.4f}, top={paras[:,3].mean():.4f}, close={paras[:,5].mean():.4f}")
            paras = paras.copy()
            paras[:, 1] *= f["rtop"]
            paras[:, 3] *= f["top"]
            paras[:, 5] *= f["close"]
            print(f"  校正後 mean sigma → rtop={paras[:,1].mean():.4f}, top={paras[:,3].mean():.4f}, close={paras[:,5].mean():.4f}")
            data_name_suffix = "_sigmaCorrected"

        # ── 停損機制實驗（拿掉停損，只保留到期強制平倉/正常平倉）──────────
        if hyperparameters.disable_stop_loss:
            print("[Disable Stop Loss] Tsl 將被覆寫成 float64 最大值，停損不會被觸發")
            data_name_suffix += "_noStopLoss"

        backtest_env = Backtest(None,
                                test,
                                open_prob_threshold=args.open_prob_threshold,
                                mode=args.Loss,
                                experiment_file=experiment_file,
                                must_open=args.must_open,
                                do_copula_model=args.do_copula_model,
                                sl_offset=args.sl_offset,
                                sl_sigma=args.sl_sigma,
                                do_dynamic_sl=args.do_dynamic_sl,
                                data_name=f"Test_sl_sigma={args.sl_sigma}_Loss={args.Loss}{data_name_suffix}",
                                do_discard_expect_prof_threshold=args.do_discard_expect_prof_threshold,
                                expect_prof_threshold=args.expect_prof_threshold,
                                use_adj_expect_prof=args.use_adj_expect_prof,
                                args=args,
                                split_info=split_info,
                                baseline_mode=False,
                                disable_stop_loss=hyperparameters.disable_stop_loss)
        records_df, pred_paras, open_prob, cor_mat_out = backtest_env.trade(
            existed_paras=paras,
            existed_revert_prob=revert_prob,
            existed_cor_mat=cor_mat
        )
        backtest_env.summary()

        # inference_cache 保留不刪除：同一份 GPU inference 結果可以重複拿來測試
        # 不同的 backtest 參數組合（例如 must_open / do_discard_expect_prof_threshold），
        # 不需要每次都重新跑一次 GPU inference。
        # 若要強制重新產生 cache，手動刪除 inference_dir 或改回 mode="inference_only" 覆蓋即可。
        print(f"Inference cache 保留：{inference_dir}（未刪除，可供下次 backtest_from_saved 重複使用）")

    elif args.baseline_mode:
        # 1.5σ 固定門檻 baseline：不需要 DL 模型，直接回測
        backtest_env = Backtest(None,
                                test,
                                mode=args.Loss,
                                experiment_file=experiment_file,
                                do_copula_model=False,
                                sl_offset=None,
                                sl_sigma=None,
                                do_dynamic_sl=False,
                                data_name="Test_Baseline_1.5sigma",
                                args=args,
                                split_info=split_info,
                                baseline_mode=True)
        records_df, pred_paras, open_prob, cor_mat = backtest_env.trade()
        backtest_env.summary()
    elif not args.do_constant_output:
        # init model
        PT_model = model.GGPTS(args).to(args.device)
        #PT_model = torch.compile(PT_model) # 模型編譯成更優化的底層指令，加速
        if model_path:
            print(f"Pretrained model loaded: {model_path}")
            PT_model.load_model(model_path)

        if mode == "train":
            PT_model.train_model(train_data=train_data,
                                 train_labels=train_labels,
                                 val_data=val_data,
                                 val_labels=val_labels,
                                 do_test=True,
                                 test_data=test_data,
                                 test_labels=test_labels,
                                 resume_ckpt_path=model_path)
            PT_model.plot_train_history(f"Loss={args.Loss}", experiment_file=experiment_file)
            print(f"Load best validation model {PT_model.best_model_name}")
            PT_model.load_model(PT_model.best_model_name)
        elif mode == "test":
            if PT_model.best_model_name and os.path.exists(PT_model.best_model_name):
              PT_model.load_model(PT_model.best_model_name)  # 載入 best model
            else:
              print("[Test] best model not found, using: {model_path}")  # fallback
        else:
            raise ValueError(f"Wrong mode '{mode}'! For DL model: 'train' or 'test'. For split GPU/CPU: use 'inference_only' then 'backtest_from_saved'.")

        #test_loss = PT_model.evaluate(test_data, test_labels)
        #print(f"Test set loss : {test_loss.item():5f}")

        # backtest
        backtest_env = Backtest(PT_model,
                         test,
                         open_prob_threshold=args.open_prob_threshold,
                         mode=args.Loss,
                         experiment_file=experiment_file,
                         must_open=args.must_open,
                         do_copula_model=args.do_copula_model,
                         sl_offset=args.sl_offset,
                         sl_sigma=args.sl_sigma,
                         do_dynamic_sl=args.do_dynamic_sl,
                         data_name=f"Test_sl_sigma={args.sl_sigma}_Loss={args.Loss}",
                         do_discard_expect_prof_threshold=args.do_discard_expect_prof_threshold,
                         expect_prof_threshold=args.expect_prof_threshold,
                         use_adj_expect_prof=args.use_adj_expect_prof,
                         args=args,
                         split_info=split_info,
                         baseline_mode=False)
        records_df, pred_paras, open_prob, cor_mat = backtest_env.trade()
        backtest_env.summary()

        # ── 同步寫入 inference_cache/，讓 train/test 跑完後可以直接用
        #    mode="backtest_from_saved" 重跑回測（例如套用 post-hoc sigma 校正），
        #    不必再手動跑一次 inference_only 重新產生 cache ──
        os.makedirs(inference_dir, exist_ok=True)
        np.save(inference_dir + "paras.npy",       pred_paras)
        np.save(inference_dir + "revert_prob.npy", open_prob)
        np.save(inference_dir + "cor_mat.npy",     cor_mat)
        test.to_pickle(inference_dir + "test_df.pkl")
        print(f"Inference cache 同步更新 → {inference_dir}")
    else:
        # constant parameters
        backtest_env = Backtest(None,
                                test,
                                open_prob_threshold=args.open_prob_threshold,
                                mode=args.Loss,
                                experiment_file=experiment_file,
                                must_open=args.must_open,
                                do_copula_model=args.do_copula_model,
                                sl_offset=args.sl_offset,
                                sl_sigma=args.sl_sigma,
                                do_dynamic_sl=args.do_dynamic_sl,
                                data_name=f"Test_sl_sigma={args.sl_sigma}_Loss={args.Loss}",
                                do_discard_expect_prof_threshold=args.do_discard_expect_prof_threshold,
                                expect_prof_threshold=args.expect_prof_threshold,
                                use_adj_expect_prof=args.use_adj_expect_prof,
                                args=args
                                )
        global_paras = args.global_paras['rtop'] + args.global_paras['top'] + args.global_paras['close']

        existed_pred_paras = np.tile(global_paras, (len(test), 1))
        existed_revert_prob = np.tile([0.0,1.0], (len(test), 1))
        global_cor_mat = args.global_paras['cormat']
        existed_cor_mat = np.tile(global_cor_mat, (len(test), 1))
        records_df, pred_paras, open_prob, cor_mat = backtest_env.trade(existed_pred_paras,
                                                                        existed_revert_prob,
                                                                        existed_cor_mat)
        backtest_env.summary()


if __name__ == '__main__':
    main_time = utils.get_time()
    # Fix: find_latest_checkpoint returns an absolute path already,
    # so prepending PathRoot would double it.
    # For backward compatibility, still prepend PathRoot if path is relative.
    _model_path = args.model_path
    if _model_path is not None and not os.path.isabs(_model_path):
        _model_path = PathRoot + _model_path
    main(args, split_info,
         mode=args.mode,
         model_path=_model_path,
         main_time=main_time
         )
