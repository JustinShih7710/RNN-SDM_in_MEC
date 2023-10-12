import pandas as pd

dtypes = {'data_timestep': int, 'vehicle_id': str, 'vehicle_x': float, 'vehicle_y': float}
veh = pd.read_csv('veh.csv', dtype=dtypes)

MEC_server = pd.read_json('MEC_server.json')

rows = []
for timestep in range(274, 899, 1):
    temp = veh.query('data_timestep == @timestep').sort_values(by=['vehicle_id'])
    for veh_temp in temp[['vehicle_id', 'vehicle_x', 'vehicle_y']].values.tolist():
        for mec_temp in MEC_server[["id", "x", "y"]].values.tolist():
            D_VtoS = (abs(veh_temp[1] - mec_temp[1]) ** 2 + abs(veh_temp[2] - mec_temp[2]) ** 2) ** 0.5
            if D_VtoS <= 100.0:
                rows.append(
                    {"data_timestep": timestep, "mec_id": mec_temp[0], 'veh_id': veh_temp[0], 'distance': D_VtoS})


timeslot_mec_content = pd.DataFrame(rows, columns=['data_timestep', 'mec_id', 'veh_id', 'distance'])

timeslot_mec_content.to_csv("timeslot_mec_content.csv", index=False)
