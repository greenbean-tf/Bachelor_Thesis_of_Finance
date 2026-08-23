import pandas as pd
import numpy as np
import os
import pickle
from tqdm.auto import tqdm
from hyperparameters import PathRoot, preprocess_data_path, Tax
from hyperparameters import open_delete, close_delete, formation_period, trading_period
from hyperparameters import formation_table_dir
from hyperparameters import start_year, start_month, end_year, end_month
import utils
import json

# Return a dataframe of all pairs, which
# includes  pair info(date, stock1_name, stock2_name),
#           model input(spread, stock_return1, stock_return2),
#           labels(do_revert, Norm_Rtop ,Norm_Top, Norm_Close, Norm_Tax)
#
# 逐年處理＋逐年存checkpoint（存在 save_path+"_checkpoints/" 底下）。中斷後重跑
# 會自動跳過已經有checkpoint的年份，不用從頭重算整個year range（年份內部仍是
# 逐月抓資料/計算，只是checkpoint的存檔粒度是一整年，不是每個月都存一次）。
def create_preprocess_data(start_year=2015, start_month=1, end_year=2019, end_month=12, save_path=PathRoot):

    checkpoint_dir = save_path + "_checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    years = list(range(start_year, end_year + 1))
    yearly_dfs = []

    for year in tqdm(years, desc="整體進度", unit="year"):
        checkpoint_path = os.path.join(checkpoint_dir, f"{year}.pickle")

        if os.path.exists(checkpoint_path):
            tqdm.write(f"[Resume] {year} 已有checkpoint，跳過重算")
            year_df = pd.read_pickle(checkpoint_path)
        else:
            # 第一年/最後一年可能不是完整的1~12月（例如 start_month=10 代表
            # 這一年只從10月算起），中間年份則一律處理全年1~12月
            m_start = start_month if year == start_year else 1
            m_end = end_month if year == end_year else 12
            month_dfs = [_build_month_preprocess_data(year, m)
                         for m in tqdm(range(m_start, m_end + 1), desc=f"{year}月份", unit="month", leave=False)]
            year_df = pd.concat(month_dfs, ignore_index=True) if month_dfs else pd.DataFrame()

            # 先寫暫存檔再原子性覆蓋，避免中途被中斷時留下寫一半的checkpoint
            # 被誤判為「已完成」而永遠不會被補算
            tmp_path = checkpoint_path + ".tmp"
            year_df.to_pickle(tmp_path)
            os.replace(tmp_path, checkpoint_path)
            tqdm.write(f"[存檔] {year} 完成，{len(year_df)} 筆樣本，checkpoint已儲存")

        yearly_dfs.append(year_df)

    df = pd.concat(yearly_dfs, ignore_index=True) if yearly_dfs else pd.DataFrame()
    df.reset_index(drop=True, inplace=True)
    df.to_pickle(save_path)
    print(f"\n全部完成，共 {len(df)} 筆樣本，已存到：{save_path}")


# 處理單一個月份：讀分鐘價格、讀formation table算spread、跑DataPreprocess，
# 回傳整理好的DataFrame（每個月是一個獨立的checkpoint單位）
def _build_month_preprocess_data(y, m):
    min_stock_data = get_min_stock_data(start_year=y, start_month=m, end_year=y, end_month=m)
    FormationTable_Spread = get_FormationTable_FTSpread(start_year=y, start_month=m, end_year=y, end_month=m,
                                                          spread_forming_stock_data=min_stock_data)
    data, labels, pairs_info = DataPreprocess(FormationTable_Spread, min_stock_data)

    Norm_Spread = [d.tolist() for d in data[:, :, 0]]
    S1_Return = [d.tolist() for d in data[:, :, 1]]
    S2_Return = [d.tolist() for d in data[:, :, 2]]

    df = pd.DataFrame({
        "Date": pairs_info[:, 2],
        "S1": pairs_info[:, 0],
        "S2": pairs_info[:, 1],
        "Norm_Spread": Norm_Spread,
        "S1_Return": S1_Return,
        "S2_Return": S2_Return,
        "Revert": labels[:, 0],
        "Norm_Rtop": labels[:, 1],
        "Norm_Top": labels[:, 2],
        "Norm_Close": labels[:, 3],
        "Norm_Tax": labels[:, 4],
    })

    df['Date'] = df['Date'].apply(int)
    df['Norm_Spread'] = df['Norm_Spread'].apply(np.array)
    df['S1_Return'] = df['S1_Return'].apply(np.array)
    df['S2_Return'] = df['S2_Return'].apply(np.array)
    df = df.dropna()
    df = df[~np.isinf(df["Norm_Rtop"])]
    df = df[~np.isinf(df["Norm_Top"])]
    df = df[~np.isinf(df["Norm_Close"])]
    df = df[~np.isinf(df["Norm_Tax"])]
    df.reset_index(drop=True, inplace=True)
    return df


# return dict of : { "date" : stockDataFrame[row="min stock price", col="stock_names"]}
def get_min_stock_data(start_year=2015, start_month=1, end_year=2019, end_month=12):
    stock_data = dict()
    nb_months = (end_month - start_month) + 12 * (end_year - (start_year)) + 1
    y = start_year
    m = start_month
    folder = PathRoot + "full_data_AB"
    for i in range(nb_months):
        # 直接列出資料夾裡屬於這個月份的檔案，而不是逐一嘗試1~31號猜檔名存不存在
        prefix = f"{y}-{m:02d}-"
        month_files = sorted(f for f in os.listdir(folder) if f.startswith(prefix) and f.endswith("_AB.csv"))
        for fname in tqdm(month_files, desc=f"讀取分鐘股價 {y}/{m:02d}", unit="day", leave=False):
            date_str = fname[:10].replace('-', '')
            stock_data[date_str] = pd.read_csv(os.path.join(folder, fname))
        m += 1
        if m > 12:
            m = 1
            y += 1
    return stock_data

# return Dataframe of col=["stock1", "stock2", "model_type", "mu", "stdev", "spread(formation+trading)", "w1", "w2", "date"]
# of data from start_year(default=2015) to  end_year(default=2018)
def get_FormationTable_FTSpread(start_year=2015, start_month=1, end_year=2018, end_month=12,spread_forming_stock_data=None):

    EMPTY_COLUMNS = ["stock1", "stock2", "model_type", "mu", "stdev", "date", "spread"]

    nb_months = (end_month - start_month) + 12 * (end_year - (start_year)) + 1
    df_list = []
    y = start_year
    m = start_month
    folder = formation_table_dir + '/'
    suffix = f"for{formation_period}del{open_delete}_AB.csv"
    # for each month
    for i in range(nb_months):
        df_daily_list = []
        # 直接列出這個月份實際存在的formation table檔案，而不是逐一嘗試1~31號猜檔名存不存在
        prefix = f"{y}{m:02d}"
        month_files = sorted(f for f in os.listdir(folder) if f.startswith(prefix) and f.endswith(suffix))
        # for each day in a month
        for fname in tqdm(month_files, desc=f"讀取formation table {y}/{m:02d}", unit="day", leave=False):
            d = int(fname[6:8])
            formation_table_path = os.path.join(folder, fname)
            daily_df = pd.read_csv(formation_table_path).loc[:, ["S1", "S2", "VECM_Model_Type", "Johansen_intercept", "Johansen_std", "w1", "w2"]]
            daily_df.drop(daily_df[daily_df["Johansen_std"]<=0].index, inplace=True)
            daily_df.reset_index(inplace=True)
            daily_df["date"] = [int(f"{y}{m:02d}{d:02d}") for _ in range(len(daily_df))]
            daily_spread_list = []
            daily_delete_list = []
            # for each pair in a day
            # 注意：這裡故意不對逐配對迴圈加tqdm——單日配對數常上萬筆，若輸出被導向
            # 檔案（例如Colab用!python執行、或log重導向），tqdm沒辦法用\r覆蓋同一行，
            # 每次刷新都會變成新的一行，容易洗出成千上萬行。日期層級的bar（見上面
            # month_files那層）已經提供夠用的進度資訊，不需要再往下細分到配對層級。
            for i in range(len(daily_df)):
                pair = daily_df.iloc[i]
                s1, s2 = pair.S1, pair.S2
                w1, w2 = pair.w1, pair.w2
                spread = np.zeros(formation_period + trading_period)
                try:
                    date_stock_prices = spread_forming_stock_data[f"{y}{m:02}{d:02}"]
                    pair_stock_prices = date_stock_prices[[s1, s2]]
                    p = np.array(pair_stock_prices).T
                    p1, p2 = p[0], p[1]
                    if len(p1) != len(p2):
                        raise ValueError(f"p1/p2長度不一致 ({len(p1)} vs {len(p2)})")

                    if len(p1) >= open_delete+formation_period+trading_period and not (p1==0).any() and not (p2==0).any():
                        spread = (w1 * np.log(p1) + w2 * np.log(p2))[open_delete:open_delete+formation_period + trading_period]
                    elif formation_period + trading_period <= len(p1) and len(p1) < open_delete+formation_period + trading_period and not (p1==0).any() and not (p2==0).any():
                        spread = (w1 * np.log(p1) + w2 * np.log(p2))[-(formation_period+trading_period):]
                    else:
                        daily_delete_list.append(i)

                    if len(spread) != formation_period+trading_period:
                        raise ValueError(f"spread長度不對 ({len(spread)})")
                except Exception as e:
                    # 單一配對計算失敗（例如資料長度異常）不中斷整個月份的處理，
                    # 當作捨棄該配對處理，並印出來方便事後排查
                    tqdm.write(f"  [配對失敗] {y}{m:02d}{d:02d} {s1}-{s2}：{repr(e)}")
                    spread = np.zeros(formation_period+trading_period)
                    if i not in daily_delete_list:
                        daily_delete_list.append(i)
                daily_spread_list.append(spread)
            daily_df["spread"] = daily_spread_list
            daily_df.reset_index(drop=True, inplace=True)
            daily_df = daily_df.drop(index=daily_delete_list)
            # w1/w2已經用來算過spread，後面沒有任何地方會再讀這兩欄，用完就丟避免留下容易誤讀的死欄位
            daily_df = daily_df.drop(columns=["w1", "w2"])
            df_daily_list.append(daily_df)

        if df_daily_list:
            df_month = pd.concat(df_daily_list)
            df_month = df_month.rename(columns={"S1": "stock1", "S2": "stock2", "VECM_Model_Type": "model_type",
                    "Johansen_intercept": "mu", "Johansen_std": "stdev"})
        else:
            df_month = pd.DataFrame(columns=EMPTY_COLUMNS)
        df_list.append(df_month)

        m += 1
        if m > 12:
            m = 1
            y += 1

    df = pd.concat(df_list) if df_list else pd.DataFrame(columns=EMPTY_COLUMNS)
    if len(df) > 0:
        df = df[df["model_type"]<3.1]  # only model 1,2, and 3
        df['date'] = df['date'].apply(int)
    return df


# Data Preprocessing

# df = Dataframee of (data size x col=["stock1", "stock2", "model_type", "mu", "stdev", "spread(formation+trading)", "w1", "w2", "date"])
# output[0] = Model Input data( [formation period x 3 (Spread, Stock1 price return, Stock2 price return)] )
#          (shape = (data_size, formation_period, 3))
# output[1] = Model Labels data( [data size][Revert, Top, Rtop, Close, NormTax]
#          (shape = (data_size, 5)
# output[2] = pairs infomation : [s1_name, s2_name, date]
#          (shape = (data_size, 3))
def DataPreprocess(df, min_stock_data):

    data_list = []
    labels_list = []
    pairs_info = []

    # mininterval限制刷新頻率（同一個月常有數萬到十幾萬筆樣本），避免輸出被導向
    # 檔案時因為沒辦法用\r覆蓋同一行，每次刷新都變成新的一行洗版
    for i in tqdm(range(len(df)), desc="Preprocessing", unit="pair", leave=False, mininterval=1):
        '''-------------------------Model Input data--------------------'''
        date = str(df.iloc[i].date)
        s1 = str(df.iloc[i].stock1)
        s2 = str(df.iloc[i].stock2)

        if len(np.array(df.iloc[i].spread)) != (formation_period+trading_period):
            tqdm.write(f"spread of {{{date} {s1} {s2}}}, spread length: {len(np.array(df.iloc[i].spread))}")
            continue

        # Stock data is between open_delete:open_delete+formation_period+trading_period
        if len(min_stock_data[date][s1].to_numpy()) < 266 :
            if len(min_stock_data[date][s1].to_numpy()) == 265:
                # special case in Taiwan stock
                min_form_s1_price = min_stock_data[date][s1].to_numpy()[15:165]
                min_form_s2_price = min_stock_data[date][s2].to_numpy()[15:165]
            else:
                tqdm.write(f"min data {date} {s1} {s2} {len(min_stock_data[date][s1].to_numpy())}")
                continue
        else:
            min_form_s1_price = min_stock_data[date][s1].to_numpy()[open_delete:open_delete + formation_period]
            min_form_s2_price = min_stock_data[date][s2].to_numpy()[open_delete:open_delete + formation_period]

        formation_period_spread = np.array((df.iloc[i].spread).copy())[:formation_period]
        spread_mean = df.iloc[i].mu
        spread_std = df.iloc[i].stdev
        formation_period_norm_spread = (formation_period_spread - spread_mean) / spread_std  # norm by formation period mu and std
        formation_period_absNorm_spread = np.abs(formation_period_norm_spread)

        # only take the data of spreads that at least touched mean once
        if utils.formation_spread_touch_mean_once(formation_period_spread, spread_mean):
            pairs_info.append([s1, s2, date])
        else:
            continue

        # Transform formation min stock price into stock return
        min_form_s1_price_return = utils.price_to_return(min_form_s1_price)
        min_form_s2_price_return = utils.price_to_return(min_form_s2_price)

        d = np.expand_dims(np.vstack((formation_period_norm_spread, min_form_s1_price_return, min_form_s2_price_return)).transpose(), axis=0)
        data_list.append(d)


        '''-------------------------Model Labels data--------------------'''
        trading_period_spread = np.array(df.iloc[i].spread.copy())[formation_period:formation_period+trading_period]
        trading_period_norm_spread = (trading_period_spread - spread_mean) / spread_std  # norm by formation period mu and std

        trading_period_absNorm_spread = np.abs(trading_period_norm_spread)
        Norm_Top = np.max(trading_period_absNorm_spread)
        Norm_Rtop = utils.find_PRtop(trading_period_norm_spread)
        if Norm_Rtop == -1:
            Revert = 0
        else:
            Revert = 1
        Norm_Close = trading_period_absNorm_spread[-1]
        Norm_Tax = Tax / spread_std

        l = np.empty(5)
        l[0] = Revert
        l[1] = Norm_Rtop
        l[2] = Norm_Top
        l[3] = Norm_Close
        l[4] = Norm_Tax
        l = np.expand_dims(l, axis=0)

        labels_list.append(l)

    # concat leftover tmp_data and tmp_labels to data and labels
    # 逐月處理後，單一個月完全沒有合格配對是有可能發生的（例如整月資料異常），
    # 這裡要避免對空list做concatenate而crash
    if data_list:
        data = np.concatenate(data_list, axis=0)
        labels = np.concatenate(labels_list, axis=0)
        pairs_info_arr = np.array(pairs_info)
    else:
        data = np.empty((0, formation_period, 3))
        labels = np.empty((0, 5))
        pairs_info_arr = np.empty((0, 3), dtype=object)

    return data, labels, pairs_info_arr


# return (train_data, val_data, test_data)
def Split_data(preprocess_data, test_size=0.2, val_size=0.1, by_time=False, split_dates=None):

    # Time dependence split

    if by_time:
        train_idx = ((preprocess_data["Date"]>=split_dates["train_start"]) & (preprocess_data["Date"]<=split_dates["train_end"]))
        val_idx = (preprocess_data["Date"]>=split_dates["validate_start"]) & (preprocess_data["Date"]<=split_dates["validate_end"])
        test_idx = (preprocess_data["Date"]>=split_dates["test_start"]) & (preprocess_data["Date"]<=split_dates["test_end"])
        train_data = preprocess_data[train_idx]
        val_data = preprocess_data[val_idx]
        test_data = preprocess_data[test_idx]

        print("Training data date:")
        print(f"    From {np.min(train_data['Date'])} to {np.max(train_data['Date'])}")
        print("Validation data date:")
        print(f"    From {np.min(val_data['Date'])} to {np.max(val_data['Date'])}")
        print("Testing data date:")
        print(f"    From {np.min(test_data['Date'])} to {np.max(test_data['Date'])}")
    else:
        train_data = preprocess_data.iloc[:int(len(preprocess_data) * (1 - test_size - val_size))]
        val_data = preprocess_data.iloc[int(len(preprocess_data) * (1 - test_size - val_size)):int(len(preprocess_data) * (1 - test_size))]
        test_data = preprocess_data.iloc[int(len(preprocess_data) * (1 - test_size)):]

    print("All data length       :", len(preprocess_data))
    print("Train data length     :", len(train_data))
    print("Validation data length:", len(val_data))
    print("Test data length      :", len(test_data))

    return train_data, val_data, test_data


def extract_cluster_data(cluster_info_path, data, cluster_n, n_components, type, split_dict):

    path = cluster_info_path + f'{type}_{str(split_dict[f"{type}_start"])[:6]}_to_{str(split_dict[f"{type}_end"])[:6]}'\
                                     f'_train_on_{str(split_dict[f"train_start"])[:6]}_to_{str(split_dict[f"train_end"])[:6]}'\
                                     f'_{n_components}cluster_.json'
    with open(path, 'r') as f:
        cluster_info = json.load(f)
        cluster_type_list = []
        for i in range(len(data)):
            cluster_type_list.append(cluster_info[f"{data.iloc[i].Date}_{data.iloc[i].S1}_{data.iloc[i].S2}"])
        data["cluster_type"] = np.array(cluster_type_list)
    return data[data["cluster_type"]==cluster_n]


if __name__ == "__main__":
    # 直接用hyperparameters.py目前作用中的test_case設定的日期範圍跟輸出路徑，
    # 跟main.py讀取preprocess_data_path時預期的檔名（preprocess_data_{start_year}_
    # {start_month}_to_{end_year}_{end_month}.pickle）保持一致，不用另外傳參數。
    create_preprocess_data(start_year=start_year, start_month=start_month,
                            end_year=end_year, end_month=end_month,
                            save_path=preprocess_data_path)
