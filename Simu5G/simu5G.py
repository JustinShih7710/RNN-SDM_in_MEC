import pandas as pd
from MEC import MEC
from UE import UE
import random
import tensorflow as tf
import numpy as np
import gc

a = 0.4
b = 0.3
c = 0.3
p = 0.4
MAX_DISTANCE = 166.7 # 500m => 3*3 166.7*166.7 ;=> 2*2 250.0*250.0


def loss_function(loss, work_finished, resource_utilize):
    return a * loss - b * work_finished - c * resource_utilize


def read_data():
    dtypes = {'data_timestep': int, 'vehicle_id': str, 'vehicle_x': float, 'vehicle_y': float}
    veh = pd.read_csv('veh.csv', dtype=dtypes)

    cols = ['data_timestep', 'mec_id', 'veh_id']
    dtypes = {'data_timestep': int, 'mec_id': str, 'veh_id': str}
    veh_mec = pd.read_csv('Result.csv', dtype=dtypes, usecols=cols)

    mec = pd.read_json('MEC_server.json', orient='table')
    return veh, veh_mec, mec


def random_zero_one(p):
    if random.random() < p:
        return True
    else:
        return False


def transfer_veh(vehicle_location):
    if vehicle_location.shape == (1, 2):
        vehicle_location = vehicle_location / 1000.0
        vehicle_location = vehicle_location.reshape(vehicle_location.shape[0], vehicle_location.shape[1], 1)
    elif vehicle_location.shape == (1, 2, 1):
        vehicle_location = vehicle_location.reshape(vehicle_location.shape[0], vehicle_location.shape[1])
        vehicle_location = vehicle_location * 1000.0
    return vehicle_location


def communication(veh_id, ue_dict, mec_dict):
    global mec
    vehicle_xy = ue_dict[veh_id].get_location()
    for mec_index in range(0, len(mec), 1):
        mec_id = mec[mec_index]
        mec_xy = mec_dict[mec_id].get_location()
        D_VtoS = ((vehicle_xy[0][0] - mec_xy[0][0]) ** 2 + (
                vehicle_xy[0][1] - mec_xy[0][1]) ** 2) ** 0.5
        if D_VtoS <= MAX_DISTANCE:
            if ue_dict[veh_id].get_mec_id() != mec_id:
                tmp = ue_dict[veh_id].get_mec_server()
                tmp.append(mec_id)
                ue_dict[veh_id].set_mec_server(tmp)
                tmp = mec_dict[mec_id].get_task()
                tmp.append(veh_id)
                mec_dict[mec_id].set_task(tmp)


def initial_communication(veh_id, ue_dict, mec_dict):
    global mec
    date_rate = random.randint(1500000, 1750000)  # bits
    ue_dict[veh_id].set_task_bit_rate(date_rate)
    vehicle_xy = ue_dict[veh_id].get_location()
    # query MEC
    rows = []
    for mec_id in mec:
        mec_xy = mec_dict[mec_id].get_location()
        D_VtoS = ((vehicle_xy[0][0] - mec_xy[0][0]) ** 2 + (
                vehicle_xy[0][1] - mec_xy[0][1]) ** 2) ** 0.5
        if D_VtoS <= MAX_DISTANCE:
            rows.append({"mec_id": mec_id, 'veh_id': veh_id, 'distance': D_VtoS})
    df_cols = ['mec_id', 'veh_id', 'distance']
    mec_veh = pd.DataFrame(rows, columns=df_cols).sort_values(['distance'])

    mec_id = mec_veh['mec_id'].values[0]
    # UE send request to MEC
    tmp = ue_dict[veh_id].get_mec_server()
    tmp.append(mec_id)
    ue_dict[veh_id].set_mec_server(tmp)
    ue_dict[veh_id].set_mec_id(mec_id)
    tmp = mec_dict[mec_id].get_task()
    tmp.append(veh_id)
    mec_dict[mec_id].set_task(tmp)

    del date_rate, mec_veh, rows, vehicle_xy, mec_id


def migration(veh_id, work_finished, resource_utilize):
    global mec, ue_dict, mec_dict, model
    # 獲取當前位置
    vehicle_xy = ue_dict[veh_id].get_location()
    vehicle_xy = transfer_veh(vehicle_xy)
    # RNN 預測新位置
    vehicle_xy_next_rnn = np.array(model(vehicle_xy, training=False))
    tf.keras.backend.clear_session()
    vehicle_xy_next_rnn = transfer_veh(vehicle_xy_next_rnn)
    # 移動下一個MEC工作 start
    rows = []
    rows_rnn = []
    vehicle_xy_next = ue_dict[veh_id].get_next_location()
    if vehicle_xy_next[0][0] == 0 and vehicle_xy_next[0][1] == 0:
        ue_dict[veh_id].reset()
        tmp = ue_dict[veh_id].get_task_list()
        tmp.append(False)
        ue_dict[veh_id].set_task(tmp)
        return

    # 搜尋MEC server
    for mec_id in mec:
        mec_xy = mec_dict[mec_id].get_location()
        D_VtoS = ((vehicle_xy_next[0][0] - mec_xy[0][0]) ** 2 + (
                vehicle_xy_next[0][1] - mec_xy[0][1]) ** 2) ** 0.5
        D_VtoS_RNN = ((vehicle_xy_next_rnn[0][0] - mec_xy[0][0]) ** 2 + (
                vehicle_xy_next_rnn[0][1] - mec_xy[0][1]) ** 2) ** 0.5
        if D_VtoS <= MAX_DISTANCE:
            rows.append({"mec_id": mec_id, 'veh_id': veh_id, 'distance': D_VtoS})
        if D_VtoS_RNN <= MAX_DISTANCE:
            rows_rnn.append({"mec_id": mec_id, 'veh_id': veh_id, 'distance': D_VtoS_RNN})
    mec_veh = pd.DataFrame(rows, columns=['mec_id', 'veh_id', 'distance']).sort_values(['distance'])
    mec_veh_rnn = pd.DataFrame(rows_rnn, columns=['mec_id', 'veh_id', 'distance']).sort_values(['distance'])
    mec_id_final = ''
    for mec_id in mec_veh['mec_id']:
        for mec_id_rnn in mec_veh_rnn['mec_id']:
            if mec_id == mec_id_rnn:
                mec_id_final = mec_id
                break

    # 移動下一個MEC工作 end
    # 判斷是否一致
    if len(mec_id_final) != 0:
        mec_id = mec_id_final
        # 改新的mec
        ue_dict[veh_id].set_mec_id(mec_id)
        ue_dict[veh_id].clr_mec_server()
        tmp = ue_dict[veh_id].get_mec_server()
        tmp.append(mec_id)
        ue_dict[veh_id].set_mec_server(tmp)

    else:  # 舊MEC中斷連線
        ue_dict[veh_id].reset()
        tmp = ue_dict[veh_id].get_task_list()
        tmp.append(False)
        ue_dict[veh_id].set_task(tmp)

    del vehicle_xy, vehicle_xy_next_rnn, rows, rows_rnn, vehicle_xy_next, mec_id, mec_veh, mec_veh_rnn


if __name__ == '__main__':

    '''gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)'''
    model = tf.keras.models.load_model('RNN_withEarlyStop.h5')
    # read data
    veh, veh_mec, mec = read_data()
    # 建立class
    # MEC
    mec_dict = {}
    for row in mec.itertuples():
        MEC_id, MEC_x, MEC_y = row.id, row.x, row.y
        mec_dict[MEC_id] = MEC(MEC_id, MEC_x, MEC_y)
    mec = list(mec_dict.keys())
    del MEC_id, MEC_x, MEC_y, row
    # UE
    ue_dict = {}
    vehicle = veh_mec.veh_id.astype('category')
    vehicle = list(vehicle.cat.categories)
    for veh_id in vehicle:
        veh_route = veh.loc[veh['vehicle_id'] == veh_id, ['data_timestep', 'vehicle_x', 'vehicle_y']].reset_index(
            drop=True)
        ue_dict[veh_id] = UE(veh_id, veh_route)
    del veh, veh_mec, veh_route
    gc.collect()
    row_mec = {'mec_id': [], 'resource_utilize': []}
    for index, mec_id in enumerate(mec):
        row_mec['mec_id'].append(mec_id)
        row_mec['resource_utilize'].append(0.0)
    for timestamp in range(274, 899, 1):
        print('timestamp : ' + str(timestamp))
        for mec_id in mec:
            mec_dict[mec_id].set_task([])
        for veh_id in vehicle:
            if ue_dict[veh_id].update_location(timestamp):
                # 判斷有無任務
                if ue_dict[veh_id].get_task_bit_rate() == 0:  # 無任務
                    if random_zero_one(p):
                        initial_communication(veh_id, ue_dict, mec_dict)
                        communication(veh_id, ue_dict, mec_dict)
                else:  # 有任務
                    if ue_dict[veh_id].get_task_bit_rate() < 0:  # 當前任務已完成
                        tmp = ue_dict[veh_id].get_task_list()
                        tmp.append(True)
                        ue_dict[veh_id].set_task(tmp)
                        ue_dict[veh_id].reset()
                        if random_zero_one(p):
                            initial_communication(veh_id, ue_dict, mec_dict)
                            communication(veh_id, ue_dict, mec_dict)
                    else:
                        mec_id = ue_dict[veh_id].get_mec_id()
                        tmp = mec_dict[mec_id].get_task()
                        tmp.append(veh_id)
                        mec_dict[mec_id].set_task(tmp)
                        communication(veh_id, ue_dict, mec_dict)
            else:
                if ue_dict[veh_id].get_task_bit_rate() < 0:  # 當前任務已完成
                    tmp = ue_dict[veh_id].get_task_list()
                    tmp.append(True)
                    ue_dict[veh_id].set_task(tmp)
                    ue_dict[veh_id].reset()
                elif ue_dict[veh_id].get_task_bit_rate() == 0:
                    continue
                else:
                    tmp = ue_dict[veh_id].get_task_list()
                    tmp.append(True)
                    ue_dict[veh_id].set_task(tmp)
                    ue_dict[veh_id].reset()

        for veh_id in vehicle:
            if ue_dict[veh_id].update_location(timestamp):
                if ue_dict[veh_id].get_task_bit_rate() > 0:
                    mec_id = ue_dict[veh_id].get_mec_id()
                    if len(ue_dict[veh_id].get_task_list()) > 0:
                        work_finished = ue_dict[veh_id].get_task_list().count(True) / len(
                            ue_dict[veh_id].get_task_list())
                    else:
                        work_finished = 0
                    resource_utilize = mec_dict[mec_id].get_resource_utilize()
                    tmp = ue_dict[veh_id].get_task_bit_rate()
                    tmp -= 800 * len(ue_dict[veh_id].get_mec_server())
                    ue_dict[veh_id].set_task_bit_rate(tmp)
                    migration(veh_id, work_finished, resource_utilize)
        for index, mec_id in enumerate(mec):
            tmp = row_mec['resource_utilize'][index]
            if tmp <= mec_dict[mec_id].get_resource_utilize():
                row_mec['resource_utilize'][index] = mec_dict[mec_id].get_resource_utilize()
        gc.collect()
    rows = []
    finish = 0
    sum_task = 0
    for veh_id in vehicle:
        temp = ue_dict[veh_id].get_task_list()
        if len(temp) != 0:
            rows.append(
                {'veh_id': veh_id, 'finished': temp.count(True), 'failed': temp.count(False), 'total': len(temp)})
        finish += temp.count(True)
        sum_task += len(temp)
    veh_finish = pd.DataFrame(rows, columns=['veh_id', 'finished', 'failed', 'total'])
    veh_finish.to_csv('RNN_3m3.csv')

    mec_resource_utilize = pd.DataFrame(row_mec, columns=['mec_id', 'resource_utilize'])
    mec_resource_utilize.to_csv('RNN_5G_3m3.csv',  index=False)

    print('Task Complete : ' + str(finish / sum_task * 100) + '%')

    # 篩選出 resource_utilize 值大於0的行
    filtered_rows = mec_resource_utilize[mec_resource_utilize['resource_utilize'] > 0]

    # 計算篩選後的 DataFrame 中每個 mec_id 的平均值
    result = filtered_rows.groupby('mec_id')['resource_utilize'].mean()

    print('Resource utilize : ' + str(result*100.0) + '%')
