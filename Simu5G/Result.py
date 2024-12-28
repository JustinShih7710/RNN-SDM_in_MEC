import pandas as pd
import function1 as fun1

dtypes = {'data_timestep': int, 'vehicle_id': str, 'vehicle_x': float, 'vehicle_y': float}
veh = pd.read_csv('veh.csv', dtype=dtypes)

dtypes = {'data_timestep': int, 'mec_id': str, 'veh_id': str, 'distance': float}
timeslot_mec_content = pd.read_csv('timeslot_mec_content.csv', dtype=dtypes)

MEC_server = pd.read_json('MEC_server.json', orient='table')
MEC_id = 'm3933'
rows = []
for i in range(274, 899, 1):
    data_timestep = i
    MEC_X = MEC_server[MEC_server.id == MEC_id].x.values[0]
    MEC_Y = MEC_server[MEC_server.id == MEC_id].y.values[0]
    temp = timeslot_mec_content[
        (timeslot_mec_content.mec_id == MEC_id) & (timeslot_mec_content.data_timestep == data_timestep)]
    for n in temp.veh_id:
        vehicle_id = n
        UE_X = veh[(veh.data_timestep == data_timestep) & (veh.vehicle_id == vehicle_id)].vehicle_x.values[0]
        UE_Y = veh[(veh.data_timestep == data_timestep) & (veh.vehicle_id == vehicle_id)].vehicle_y.values[0]
        if len(timeslot_mec_content[(timeslot_mec_content.veh_id == vehicle_id) & (
                timeslot_mec_content.data_timestep == data_timestep)]) > 1:
            D_Cell_UE = ((MEC_X - UE_X) ** 2 + (MEC_Y - UE_Y) ** 2) ** 0.5
            pL = fun1.path_loss(D_Cell_UE)
            P_received = fun1.Power_received(pL)
            if MEC_id == timeslot_mec_content[(timeslot_mec_content.veh_id == vehicle_id) & (
                    timeslot_mec_content.data_timestep == data_timestep)].mec_id.values[1]:
                MEC_id = timeslot_mec_content[(timeslot_mec_content.veh_id == vehicle_id) & (
                            timeslot_mec_content.data_timestep == data_timestep)].mec_id.values[1]
            else:
                MEC_id = timeslot_mec_content[(timeslot_mec_content.veh_id == vehicle_id) & (
                            timeslot_mec_content.data_timestep == data_timestep)].mec_id.values[0]
            MEC_X = MEC_server[MEC_server.id == MEC_id].x.values[0]
            MEC_Y = MEC_server[MEC_server.id == MEC_id].y.values[0]
            D_Cell_UE1 = ((MEC_X - UE_X) ** 2 + (MEC_Y - UE_Y) ** 2) ** 0.5
            pL1 = fun1.path_loss(D_Cell_UE1)
            P_received1 = fun1.Power_received(pL1)
            P_received += P_received1
        else:
            D_Cell_UE = ((MEC_X - UE_X) ** 2 + (MEC_Y - UE_Y) ** 2) ** 0.5
            pL = fun1.path_loss(D_Cell_UE)
            P_received = fun1.Power_received(pL)
        SINR_UE_value = fun1.SINR_UE(P_received)
        codeRate, bitnum = fun1.CQI_Table(SINR_UE_value)
        Link_rate = fun1.Link_rate(codeRate, bitnum)
        rows.append({"data_timestep": data_timestep, "mec_id": MEC_id, 'veh_id': vehicle_id, 'distance': D_Cell_UE,
                     'path loss': pL, 'Power received': P_received, 'SINR': SINR_UE_value, 'Link rate': Link_rate})
df_cols = ['data_timestep', 'mec_id', 'veh_id', 'distance', 'path loss', 'Power received', 'SINR', 'Link rate']

Result = pd.DataFrame(rows, columns=df_cols)
Result.to_csv("Result.csv", index=False)
