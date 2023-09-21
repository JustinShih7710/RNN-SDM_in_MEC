import pandas as pd

if __name__ == '__main__':
    dtypes = {'data_timestep': int, 'vehicle_id': str, 'vehicle_x': float, 'vehicle_y': float}
    veh = pd.read_csv('veh.csv', dtype=dtypes)
    dtypes = {'data_timestep': int, 'ped_id': str, 'ped_x': float, 'ped_y': float}
    ped = pd.read_csv('ped.csv', dtype=dtypes)

    ped.to_parquet('ped.parquet', engine='pyarrow', index=False)
    veh.to_parquet('veh.parquet', engine='pyarrow', index=False)
