import pandas as pd
import os
import datetime
import numpy as np
import json
import matrix_trading
import time
import wget
import glob
from shutil import copyfile
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import pickle
import configparser


# get external data
config = configparser.ConfigParser()
if os.path.isfile('config.ini'):
    config.read('config.ini')
else:
    config.read('../config.ini')
PTDQN_DATA_FOLDER = config['folder']['data_root_folder']
plt.rcParams['date.epoch'] = '0000-12-31'


# given dir_name(data of specific stock of specific year), num(specific day)
# return minuted-price table and the date
def get_stock_minute_data(dir_name, num, tick=False):
    all_data_file = os.listdir(dir_name)
    all_data_file.sort()
    in_sample_data = pd.DataFrame()
    file = all_data_file[num]
    # print(file)
    # noinspection PyBroadException
    try:
        if tick:
            df = pd.read_csv(dir_name + '/' + file, encoding='utf-8')
            df['date'] = [file[:8]] * len(df)
            return df, file[:8]
        else:
            print(dir_name + '/' + file)
            df = pd.read_csv(dir_name + '/' + file, encoding='utf-8')

        # drop last 5/4(?) min for the reason in senpai's QA, and leave last price for close price
            if len(df) == 271:
                df = df.drop([266, 267, 268, 269, 270])
            elif len(df) == 270:
                df = df.drop([266, 267, 268, 269])

            test = []
            h = 9
            m = 1
            while len(test) != len(df):
            # close price
                if len(test) == len(df) - 1:
                    test.append(datetime.datetime.strptime(file[:8], '%Y%m%d').replace(hour=13, minute=30))
            # else
                else:
                    test.append(datetime.datetime.strptime(file[:8], '%Y%m%d').replace(hour=h, minute=m))
                if m == 59:
                    m = 0
                    h = h + 1
                else:
                    m = m + 1
            df['datetime'] = test
            df['date'] = [file[:8]] * len(df)
            in_sample_data = pd.concat([in_sample_data, df], ignore_index=True, sort=True)
            return in_sample_data, file[:8]
    except:
        return pd.DataFrame(), file[:8]

# given specific date, specific year formation table folder_name, is_trend_stationary, stock1[:4627], stock2[:4627]
def get_formation_table(stock_date, data_folder, trend_stationary=False, stock1=None, stock2=None):

# get formation table csv data
    # noinspection PyBroadException
    try:

    # no formation table at that day ex: 20180108, 20180109
        if not os.path.isfile(data_folder + str(stock_date) + '_formationtable.csv'):
            print(f'{data_folder + str(stock_date)}_formationtable.csv does not exist!')
            return pd.DataFrame()


    # read formation table
        table = pd.read_csv(data_folder + str(stock_date) + '_formationtable.csv')
        print(data_folder + str(stock_date) + '_formationtable.csv')
        table.rename(columns={'S1': 'stock1', 'S2': 'stock2', 'std': 'Estd', 'Johansen_intercept': 'Emu'}, inplace=True)
        if trend_stationary:
            table = table[table['model'] > 3]
        else:
            table = table[table['model'] <= 3]
        table.dropna(inplace=True)
        table.reset_index(drop=True, inplace=True)
        table['stock1'] = table['stock1'].astype(int)
        table['stock2'] = table['stock2'].astype(int)
    except:
        print("wrong")
        table = pd.DataFrame()

    if stock1 is not None and stock2 is not None:
    # make stock info into str
        if type(stock1) == list:
            stock1 = [str(s) for s in stock1]
        else:
            stock1 = [str(stock1)]
        if type(stock2) == list:
            stock2 = [str(s) for s in stock2]
        else:
            stock2 = [str(stock2)]


    # extract the partition of formation table which the pair is indeed in (stock1, stock2)
    # Noted that stock2 > stock1 in given pair(row)
        idx = list()
        for i in range(len(table)):
            for s1, s2 in zip(stock1, stock2):
                if str(table.loc[i, 'stock1']) < str(table.loc[i, 'stock2']):
                    if s1 == str(table.loc[i, 'stock1']) and s2 == str(table.loc[i, 'stock2']):
                        idx.append(i)
                else:
                    if s2 == str(table.loc[i, 'stock1']) and s1 == str(table.loc[i, 'stock2']):
                        idx.append(i)
        table = table.loc[idx, :]
        table.reset_index(drop=True, inplace=True)

    elif stock1 is None and stock2 is None:
        pass
    elif stock1 is None or stock2 is None:
        raise ValueError('stock1 and stock2 must given')
    return table


def dump_pairs_trading_reward_json(all_actions=None, stock1=None, stock2=None, pairs_count=None,
                                   trend_stationary=False, overwrite=False, action_random=False,
                                   folder_description=None, start_year=None, end_year=None, pickle_format=False):
    """ Use given action to dump pairs trading performance to json file.

    Json file contains a list, and every element in list is `dict` format.\n
    If `all_actions=[[1.5, 2.5], [2, 3.5]]`, the dict keys will be like:\n
    {'stock1', 'stock2', 'model_type', 'mu', 'stdev', 'spread', 'w0', 'w1', 'date', '1.5,2.5', '2,3.5'}\n
    Every action key is also a dict, the key will be like:
    {'profit', 'half_tax_profit', 'zero_tax_profit', 'reward', 'record', 'close_timing', 'capital'}\n
    Hence we can use this action key to get the target action performance.

    Args:
        all_actions (list): The format of every element in the `all_actions` must be like [1.5, 25],
                            the first one `1.5` means open threshold, the second one `25` means stop-loss threshold.
                            If not given, generate actions from open threshold range [0.5, 8] and 0.05 a step,
                            stop-loss threshold [1.5, 25.5) and 1 a step, and open threshold need larger than
                            1.5 * stop-loss threshold. And add [1000000, 2500000] to represent don't open action.
                            There are total 2811 actions.
        stock1 (list): If not `None`, only use specific stock1 to dump json data.
        stock2 (list): If not `None`, only use specific stock2 to dump json data.
                        * Examples: if `stock1=[2002, 2002, 2353, 2002, 2885]` and
                        `stock2=[2887, 2892, 2892, 2353, 2892]`,  filter `2002_2887`, `2002_2892`, `2353_2892`,
                        `2002_2353`, `2885_2892` 5 pairs to dump data.
        pairs_count (int): If given, add this count to destination directory name.
        trend_stationary (bool): If `True`, only dump co-integration model 4 ~ 5, otherwise dump co-integration model
                                1 ~ 3.
        overwrite (bool): If `True`, overwrite json data in destination directory, otherwise it will check the
                   destination directory exist or not. If exists, return destination directory name directly.
        action_random (bool): If `True` and `all_action=None`, only get 300 actions randomly.
        folder_description (str): If not `None`, add this string to destination directory name.
        start_year (int): If not `None`, use this year to be the start year of dumping data. Otherwise the start year is
                            2013.
        end_year (int): If not `None`, use this year to be the end year of dumping data. Otherwise the end year is 2020.
        pickle_format (bool): If `True`, output pickle format instead of json format
    Returns:
        str: Destination directory name.

    """
# get random 300 (open, stoploss) pairs, which range : ([ 0.5 ~ 8 ], [1.5, 25.5) ),
# and add [1000000, 2500000] represent doing nothing
    folder_name = f'HsinHuaChang_reverse_'
    if all_actions is None:
        np.random.seed(555)
        all_actions = list()
        for open_threshold in np.linspace(0.5, 8, 151):
            for stop_loss_threshold in np.linspace(1.5, 25.5, 25)[:-1]:
                if stop_loss_threshold > 1.5 * open_threshold:
                    all_actions.append([open_threshold, stop_loss_threshold])
        if action_random:
            idx = np.random.choice(len(all_actions), 299, replace=False)
            all_actions = list(np.array(all_actions)[idx])
            folder_name += 'random_'
        all_actions = [list(x) for x in all_actions]
        all_actions.insert(len(all_actions), [1000000, 2500000])
        # reverse for reason in senpai's QA
        all_actions.reverse()
    folder_name += f'{len(all_actions)}_'
    tick = False
    if tick:
        raise ValueError('Not fully implement using every 5 second tick data to dump the pairs trading performance!\n'
                         'Please set `tick=False` instead.')


# parameters
    method = None
    reward_type = None
    output_stock_price = False
    dump_spread_plot = False
    formation_time = 150
    trade_time = 100
    transaction_cost = 0.0015
    transaction_cost_threshold = 0.0015
    maxi = 5
    capital = 50000000
    adf = False
    fore_lag5 = False
    new_std = True
    dt = ''
    cost_gate_type = 0

# output_file_name stuff
    if stock1 is not None and stock2 is not None:
        folder_name += f'{len(stock1)}_{len(stock2)}_'
    elif pairs_count is not None:
        folder_name += f'pairs_count={pairs_count}_'
    if method is not None:
        folder_name = f'{method}_'
    if reward_type is not None:
        folder_name += f'{reward_type}_'
    if folder_description is not None:
        folder_name += f'{folder_description}_'
    if trend_stationary:
        folder_name += f'trend_stationary_'
    if output_stock_price:
        folder_name += f'with_price_'
    target_json_data_folder = f'{PTDQN_DATA_FOLDER}/'
    if start_year is None:
        start_year = 2013
    if end_year is None:
        end_year = 2020
    raw_data_folder = 'C:/Users/user/Desktop/WeiCheCode/PTDQN-main/stock_data/data'
    # folder_name += '_trading_threshold_with_stop_loss_method3_add_stock_price'
    if new_std:
        folder_name += 'new_std'
    else:
        folder_name += 'old_std'
    if pickle_format:
        folder_name += '_pickle'
    if not overwrite:
        if not os.path.exists(target_json_data_folder + folder_name):
            os.makedirs(target_json_data_folder + folder_name)
        else:
            print(f'{folder_name} already exist!')
            return folder_name
    program_file_folder = target_json_data_folder + folder_name + '/program/'
    if not os.path.exists(program_file_folder):
        os.makedirs(program_file_folder)
    if os.getcwd().endswith('dump_json_data'):
        destination = './'
    elif os.getcwd().endswith('agent'):
        destination = '../dump_json_data/'
    else:
        destination = 'dump_json_data/'
    for file in sorted(glob.glob(f'{destination}*.py')):
        copyfile(file, program_file_folder + os.path.basename(file))
    with open(target_json_data_folder + folder_name + '/agent_action_info.json', 'w') as f:
        json.dump(all_actions, f)



# main process
    print(raw_data_folder)
    for year in range(start_year, end_year + 1):



    # folder_name_stuff
        all_dict = list()
        year_data_folder = f'{raw_data_folder}/min_data/{year}'
        year_formation_data_folder = f'{raw_data_folder}/formation_table/BIC_based_model_selection/{year}/'

        for num in range(len(os.listdir(f'{year_data_folder}/averageprice'))):
            print(f'{year_data_folder}/averageprice')



        # day1 == avg price of every minute data
        # day1_tick = last price of every minute data
            day1, day1_date = get_stock_minute_data(f'{year_data_folder}/averageprice', num)
            if tick:
                day1_tick, day1_tick_date = get_stock_minute_data(
                    f'D:/HsinHuaChangThesis/thesis/data/tick(secs)/{year}', num, tick=tick)
            else:
            # table of minuted-price in numth day and its date
                day1_tick, day1_tick_date = get_stock_minute_data(f'{year_data_folder}/minprice', num)



        # catch error
            if day1_date != day1_tick_date:
                raise ValueError(f'date {day1_date} {day1_tick_date} not match')
            if day1.empty:
                raise ValueError(f'{num}: {day1_date} is empty tick data')
            if day1_tick.empty:
                raise ValueError(f'{num}: {day1_tick_date} is empty tick data')



        # trading period data == trigger_data == last price of every minute data(exclude last 5 min)
            if not tick:
                trigger_data = day1_tick.iloc[165: 266, :]
                trigger_data.index = np.arange(0, len(trigger_data), 1)
            else:
                sec_5 = np.arange(9000 + 60 * 16, day1_tick.shape[0] - 1, 5)
                trigger_data = day1_tick.iloc[sec_5, :]
                trigger_data = trigger_data.reset_index(drop=True)



        # get date of current date
            unique_stock_date = list(np.unique(list(day1_tick['date'])))[0]
            if num == 0:
                dt = unique_stock_date[:6]
            if dt != unique_stock_date[:6]:
                if pickle_format:
                    with open(target_json_data_folder + folder_name + '/Pairs' + dt + '.pickle', 'wb') as f:
                        pickle.dump(all_dict, f)
                else:
                    with open(target_json_data_folder + folder_name + '/Pairs' + dt + '.json', 'w') as f:
                        json.dump(all_dict, f)
                dt = unique_stock_date[:6]
                all_dict = list()



        # get formation table of current date
            table = get_formation_table(unique_stock_date, year_formation_data_folder, trend_stationary, stock1, stock2)
            if table.empty:
                print(f'{num}: {unique_stock_date} is empty formation table')
                continue
            if not table.empty and (len(day1_tick) != 266):
                print(f'Data length not match: {day1_tick_date} len is {len(day1_tick)}')
                continue

        # calculate profit of all action to all pairs in formation table in current day
            daily_performance_df = pd.DataFrame()
            for i, action in zip(range(len(all_actions)), all_actions):
                print(f'calculating {i} action')
                close_times = 0     # close only when the spread return to mean(close_spread = mean + close_time = mean)
                open_times = action[0]
                stop_loss_times = action[1]
                action_name = str(open_times) + ',' + str(stop_loss_times)

            # t = trading simulator, in matrix_trading.py
                t = matrix_trading.Trading(tick, table, formation_time, trade_time, day1, trigger_data, day1_tick,
                                           open_times, close_times, stop_loss_times, maxi,
                                           transaction_cost, transaction_cost_threshold, capital,
                                           cost_gate_type=cost_gate_type, folder_name=folder_name, method=method,
                                           reward_type=reward_type, output_stock_price=output_stock_price,
                                           trend_stationary=trend_stationary, dump=dump_spread_plot)

            # df = trading profit of all pairs on current date by current action
            # daily_performance_df = df
                df = t.pairs_trading_back_test(unique_stock_date, folder_name, adf, fore_lag5, new_std)
                if len(df) == 0:
                    continue
                if output_stock_price:
                    df.columns = ['stock1', 'stock2', 'model_type', 'mu', 'stdev', 'spread', 'stock1_price',
                                  'stock2_price', action_name, 'w1', 'w2']
                else:
                    df.columns = ['stock1', 'stock2', 'model_type', 'mu', 'stdev', 'spread', action_name, 'w1', 'w2']
                df['date'] = unique_stock_date
                if daily_performance_df.empty:
                    daily_performance_df = df
                else:
                    daily_performance_df[action_name] = df[action_name]
                # assert len(daily_performance_df) == len(table), 'Len does not match!'
            all_dict += daily_performance_df.to_dict('records')


# output_formal_stuff
        if pickle_format:
            with open(target_json_data_folder + folder_name + '/Pairs' + dt + '.pickle', 'wb') as f:
                pickle.dump(all_dict, f)
        else:
            with open(target_json_data_folder + folder_name + '/Pairs' + dt + '.json', 'w') as f:
                json.dump(all_dict, f)
    return folder_name

# not used
def dump_model_type_json():
    data_folder = 'D:/new_pair_data'
    model_type_dict = dict()
    for year in range(2013, 2019):
        # year_data_folder = '{}/{}'.format(data_folder, year)
        year_formation_data_folder = '{}/newstdcompare{}/'.format(data_folder, year)
        for file_name in os.listdir(year_formation_data_folder):
            table = get_formation_table(file_name[:8], year_formation_data_folder)
            if table.empty:
                print(file_name[:8], end=':')
                print('is empty table')
                continue
            for _, row in table.iterrows():
                model_type_dict[file_name[:8] + '_' + row['stock1'] + '_' + row['stock2']] = int(row['model_type'][5])
    with open('model_type.json', 'w') as f:
        json.dump(model_type_dict, f)
# not used
def normal_json_with_twse_back_adjusted_input():
    one_min_twse = pd.read_csv('D:/ptdqn_data/1000A.TWS 1 Minute.csv')
    twse_back_adjusted = pd.read_csv('D:/ptdqn_data/daily_back_adjusted_return.csv', encoding='utf-8')
    twse_back_adjusted['ratio'] = twse_back_adjusted['報酬指數值'] / twse_back_adjusted['價格指數值']
    twse_back_adjusted['日期'] = [datetime.datetime.strptime(dt, '%Y/%m/%d') for dt in twse_back_adjusted['日期']]
    one_min_twse['Date'] = [datetime.datetime.strptime(dt, '%Y/%m/%d') for dt in one_min_twse['Date']]
    twse_adjusted = pd.DataFrame()
    start_days = 15
    day_interval = -1
    for i in range(len(twse_back_adjusted)-1, -1, -1):
        dt = twse_back_adjusted.iloc[i, 0]
        print(dt)
        ratio = twse_back_adjusted.iloc[i, -1]
        tmp = one_min_twse[one_min_twse['Date'] == dt]
        if len(tmp) == 0:
            print('pass')
            continue
        tmp['Close'] = tmp['Close'].apply(lambda x: x*ratio)
        tmp = tmp.loc[:, ['Date', 'Time', 'Close']].reset_index(drop=True)
        tmp = tmp[tmp['Time'] != '13:26:00']
        tmp = tmp[tmp['Time'] != '13:27:00']
        tmp = tmp[tmp['Time'] != '13:28:00']
        tmp = tmp[tmp['Time'] != '13:29:00']
        if len(tmp) != 266:
            raise ValueError('{} len:{} {}'.format(dt, len(tmp), i))
        for days in range(start_days, 0, day_interval):
            # noinspection PyBroadException
            try:
                pre_date = twse_back_adjusted.iloc[i + days, 0]
                tmp['{}_pre_close'.format(days)] = list(twse_adjusted[twse_adjusted['Date'] == pre_date]['Close'])
            except:
                tmp['{}_pre_close'.format(days)] = None
        twse_adjusted = twse_adjusted.append(tmp)
        twse_adjusted.reset_index(drop=True, inplace=True)
    col_name = ['Date']
    for days in range(start_days, 0, day_interval):
        twse_adjusted = twse_adjusted[[x is not None for x in twse_adjusted['{}_pre_close'.format(days)]]]
        twse_adjusted['{}_days_return'.format(days)] = \
            (twse_adjusted['Close'] - twse_adjusted['{}_pre_close'.format(days)]) / \
            twse_adjusted['{}_pre_close'.format(days)]
        col_name.append('{}_days_return'.format(days))
    twse_adjusted = twse_adjusted.loc[:, col_name]
    xx = twse_adjusted.iloc[:, 1:].values.tolist()
    twse_adjusted['15_twse_return'] = xx
    all_dict = dict()
    for dt in np.unique(twse_adjusted['Date']):
        tmp = twse_adjusted[twse_adjusted['Date'] == dt]
        tmp = tmp.iloc[16:, ]
        tmp['Date'] = [dt.strftime('%Y%m%d') for dt in tmp['Date']]
        all_dict[np.unique(tmp['Date'])[0]] = list(tmp['15_twse_return'])
    with open(f'D:/ptdqn_data/Data/20210107_{start_days}_{day_interval}_0_twse_return.json', 'w') as f:
        json.dump([all_dict], f)
#not used
def get_stock_industry_json():
    """
    data source:
    https://goodinfo.tw/StockInfo/StockList.asp?MARKET_CAT=%E4%B8%8A%E5%B8%82&INDUSTRY_CAT=%E4%B8%8A%E5%B8%82%E5%85%A8%E9%83%A8&SHEET=%E4%BA%A4%E6%98%93%E7%8B%80%E6%B3%81&SHEET2=%E6%97%A5&RPT_TIME=%E6%9C%80%E6%96%B0%E8%B3%87%E6%96%99
    """
    total_industry = ['水泥工業', '食品工業', '塑膠工業', '紡織纖維', '電機機械', '電器電纜', '生技醫療業', '化學工業', '玻璃陶瓷',
                      '造紙工業', '鋼鐵工業', '橡膠工業', '汽車工業', '電腦及週邊設備業', '半導體業', '電子零組件業', '其他電子業',
                      '通信網路業', '資訊服務業', '建材營造業', '航運業', '觀光事業', '銀行業', '保險業', '金控業', '貿易百貨業',
                      '光電業', '電子通路業', '證券業', '其他業', '油電燃氣業']
    industry_index_code = ['C01', 'C02', 'C03', 'C04', 'C05', 'C06', 'C22', 'C21', 'C08', 'C09', 'C10', 'C11', 'C12',
                           'C25', 'C24', 'C28', 'C31', 'C27', 'C30', 'C14', 'C15', 'C16', 'C17', 'C17', 'C17', 'C18',
                           'C26', 'C29', 'C17', 'C20', 'C23']
    industry_code_dict = dict()
    for name, code in zip(total_industry, industry_index_code):
        industry_code_dict[name] = code
    for c in industry_index_code:
        if not os.path.isfile('industry_index/raw_data/C01.xls'):
            wget.download(f"https://www.taiwanindex.com.tw/index/multipleDownloadExcel?s={c}&start=2000%2F01%2F22&end=",
                          f'industry_index/raw_data/{c}.xls')
            time.sleep(5)
            print(f'{c} done!')
    total_stock = pd.DataFrame()
    for i in range(31):
        df = pd.read_csv(f'stock_industry_list/StockList ({i}).csv').iloc[:, :2]
        df.iloc[:, 0] = [x.replace('=', '').replace('"', '') for x in df.iloc[:, 0]]
        df['industry'] = total_industry[i]
        total_stock = pd.concat([total_stock, df])
    print(total_stock)
    stock_industry = dict()
    for i in range(len(total_stock)):
        stock_industry[total_stock.iloc[i, 0]] = total_stock.iloc[i, 2]
    stock_industry['1704'] = '化學工業'
    stock_industry['2311'] = '半導體業'
    stock_industry['2325'] = '半導體業'
    stock_industry['2448'] = '光電業'
    stock_industry['5264'] = '電腦及週邊設備業'
    stock_industry['6452'] = '生技醫療業'
    with open(f'stock_industry.json', 'w') as f:
        json.dump(stock_industry, f)
    with open(f'industry_code.json', 'w') as f:
        json.dump(industry_code_dict, f)
#not used
def get_index_return_json():
    industry_index_code = ['C01', 'C02', 'C03', 'C04', 'C05', 'C06', 'C22', 'C21', 'C08', 'C09', 'C10', 'C11', 'C12',
                           'C25', 'C24', 'C28', 'C31', 'C27', 'C30', 'C14', 'C15', 'C16', 'C17', 'C17', 'C17', 'C18',
                           'C26', 'C29', 'C17', 'C20', 'C23', 'T00']
    industry_index_code = np.unique(industry_index_code)
    total_return_dict = dict()
    for c in industry_index_code:
        print(f'parsing {c}')
        df = pd.read_csv(f'industry_index/{c}.csv', encoding='big5', header=1)
        df['date'] = df['日期'].shift(1)
        df = df.loc[1:, :]
        df['date'] = [datetime.datetime.strptime(str(x), '%Y/%m/%d').strftime('%Y%m%d') for x in df['date']]
        col_name = ['date']
        for days in range(-25, 0):
            df[f'{days}_return'] = (df['報酬指數值'] - df['報酬指數值'].shift(days)) / df['報酬指數值'].shift(days)
            col_name.append(f'{days}_return')
        df = df.loc[1:, col_name]
        return_dict = dict()
        for dt, r in zip(df['date'], df.iloc[:, 1:].values.tolist()):
            return_dict[dt] = r
        total_return_dict[c] = return_dict
    with open(f'D:/ptdqn_data/Data/20210127_industry_index_return.json', 'w') as f:
        json.dump(total_return_dict, f)
#not used
def index_gmm_fit():
    industry_index_code = ['C01', 'C02', 'C03', 'C04', 'C05', 'C06', 'C22', 'C21', 'C08', 'C09', 'C10', 'C11', 'C12',
                           'C25', 'C24', 'C28', 'C31', 'C27', 'C30', 'C14', 'C15', 'C16', 'C17', 'C17', 'C17', 'C18',
                           'C26', 'C29', 'C17', 'C20', 'C23', 'T00']
    industry_index_code = np.unique(industry_index_code)
    daily_return = pd.DataFrame()
    daily_price = pd.DataFrame()
    for c in industry_index_code:
        df = pd.read_csv(f'industry_index/{c}.csv', encoding='big5', header=1)
        if 'date' not in daily_price.columns:
            daily_price['date'] = [datetime.datetime.strptime(str(x), '%Y/%m/%d').strftime('%Y%m%d') for x in df['日期']]
        daily_price[f'{c}_daily_price'] = df[f'報酬指數值']
        df['date'] = df['日期'].shift(1)
        df = df.loc[1:, :]
        df['date'] = [datetime.datetime.strptime(str(x), '%Y/%m/%d').strftime('%Y%m%d') for x in df['date']]
        for days in [-1]:
            df[f'{c}_daily_return'] = (df['報酬指數值'] - df['報酬指數值'].shift(days)) / df['報酬指數值'].shift(days)
        daily_return[f'{c}_daily_return'] = df[f'{c}_daily_return']
    daily_price = daily_price.loc[1:len(daily_price)-2, :]
    daily_price = daily_price.iloc[::-1].reset_index()
    daily_return = daily_return.loc[:len(daily_return)-1, :]
    daily_return = daily_return.iloc[::-1].reset_index()
    group = 3
    gmm = GaussianMixture(group, random_state=1)
    gmm.fit(daily_return)
    probs = gmm.predict_proba(daily_return)
    df_probs = pd.DataFrame(probs, index=daily_return.index,
                            columns=['probs_0', 'probs_1', 'probs_2'])
    df_probs['label'] = gmm.predict(daily_return)
    # df_probs['New label'] = df_probs['label'].replace([0, 1, 2, 3], ['Consolidate', 'Bear', 'Bull', 'Super bear'])
    for display_index in ['T00']:
        for y in ["2015", "2016", "2017", "2018"]:
            print(y)
            df_select = daily_price[[dt.startswith(y) for dt in daily_price['date']]]
            df_select['date'] = [pd.to_datetime(d, format='%Y%m%d') for d in df_select['date']]
            plt.figure(figsize=(10, 7))  # 設定圖的大小
            plt.subplots_adjust(hspace=0.5)  # 圖跟圖之間的距離
            plt.subplot(2, 1, 1)
            plt.title("GMM Clustering",
                      {'fontsize': 20})  # 設定圖標題及其文字大小
            # order = ['Super bear', 'Bear', 'Consolidate', 'Bull']  # 設定Y軸顯示順序
            df_select['content'] = [x for x in df_probs[[dt.startswith(y) for dt in daily_price['date']]]['label']]
            # plt.yticks(range(4), order, fontsize=14)  # 設定y軸刻度
            plt.scatter(df_select['date'], df_select['content'], s=9)
            plt.xticks(rotation=-15, fontsize=15)    # 設置x軸標籤旋轉角度避免重疊看不清楚
            plt.subplot(2, 1, 2)
            plt.title(f"{display_index} Index", {'fontsize': 20})  # 設定圖標題及其文字大小
            plt.yticks(fontsize=14)  # 設定y軸刻度
            plt.scatter(df_select['date'], df_select[f'{display_index}_daily_price'], s=9)
            plt.xticks(rotation=-15, fontsize=15)    # 設置x軸標籤旋轉角度
            plt.tight_layout()
            plt.show()
#not used
def get_stock_return_json():
    now_dt = datetime.datetime.now().strftime('%Y%m%d')
    folder_name = f'D:/ptdqn_data/Data/{now_dt}_stock_daily_price_return'
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    program_file = folder_name + '/program/'
    if not os.path.exists(program_file):
        os.makedirs(program_file)
    for File in sorted(glob.glob('*.py')):
        copyfile(File, program_file + os.path.basename(File))
    daily_stock_price = pd.DataFrame()
    for year in range(2012, 2020):
        year_data_folder = '{}/{}'.format('D:/new_pair_data', year)
        for num in range(len(os.listdir('{}/minprice'.format(year_data_folder)))):
            # day1 = get_stock_minute_data('averageprice', num)
            day1_tick, day1_tick_date = get_stock_minute_data('{}/minprice'.format(year_data_folder), num)
            if day1_tick.empty:
                continue
            daily_stock_price = daily_stock_price.append(day1_tick.iloc[-1, :])
    print(daily_stock_price)
    daily_stock_price.reset_index(inplace=True, drop=True)
    total_return_dict = dict()
    daily_stock_price = daily_stock_price.fillna(method='bfill')
    daily_stock_price['date'] = daily_stock_price['date'].shift(-1)
    daily_stock_price = daily_stock_price.iloc[:-1, :]
    for stock_id in daily_stock_price.columns[:-2]:
        print(f'parsing {stock_id}')
        daily_stock_price[stock_id] = [float(x) for x in daily_stock_price[stock_id]]
        # df['date'] = df['日期'].shift(1)
        col_name = ['date']
        for days in range(25, 0, -1):
            daily_stock_price[f'{days}_return'] = \
                (daily_stock_price[stock_id] - daily_stock_price[stock_id].shift(days)) / \
                daily_stock_price[stock_id].shift(days)
            col_name.append(f'{days}_return')
        df = daily_stock_price.loc[25:, col_name]
        return_dict = dict()
        for dt, r in zip(df['date'], df.iloc[:, 1:].values.tolist()):
            return_dict[dt] = r
        total_return_dict[stock_id] = return_dict
    with open(f'{folder_name}/stock_daily_price_return.json', 'w') as f:
        json.dump(total_return_dict, f)


if __name__ == '__main__':
    # dump_model_type_json()
    # get_stock_return_json()
    with open("../stock_info/HsinHuaChang_with_trend_TOP_N_STOCK1_IN_2015_2016.txt") as F:
        TOP_STOCK1 = [line.strip() for line in F]
    with open("../stock_info/HsinHuaChang_with_trend_TOP_N_STOCK2_IN_2015_2016.txt") as F:
        TOP_STOCK2 = [line.strip() for line in F]
    STOCK1 = TOP_STOCK1[:4627]
    STOCK2 = TOP_STOCK2[:4627]
    dump_pairs_trading_reward_json(overwrite=True, start_year=2015, end_year=2020, trend_stationary=False,
                                   action_random=True, pickle_format=True, stock1=STOCK1, stock2=STOCK2)
    # get_index_return_json()
    # get_stock_industry_json()
    # index_gmm_fit()
    # normal_json_with_twse_back_adjusted_input()
    # normal_json_input()
