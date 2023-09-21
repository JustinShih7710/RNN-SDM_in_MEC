import pandas as pd

if __name__ == '__main__':
    dtypes = {'data_timestep': int, 'vehicle_id': str, 'vehicle_x': float, 'vehicle_y': float}
    cols = ['data_timestep', 'vehicle_id', 'vehicle_x', 'vehicle_y']
    try:
        df = pd.read_csv('fulloutput.csv', dtype=dtypes, sep=';', usecols=cols)
    except ValueError as e:
        df = pd.DataFrame(columns=cols)
    veh = df[df['vehicle_id'].str.contains('veh|ped')]
    veh.to_csv('veh.csv', index=False)

