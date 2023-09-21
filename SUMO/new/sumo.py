import traci
import pandas as pd

if __name__ == '__main__':
    # start SUMO and connect to TraCI
    traci.start(["sumo", "-c", "osm.sumocfg"])
    columns_ped = ['data_timestep', 'ped_id', 'ped_x', 'ped_y']
    columns_veh = ['data_timestep', 'vehicle_id', 'vehicle_x', 'vehicle_y']
    data_timestep = 0
    row_ped = []
    row_veh = []
    # main simulation loop
    while data_timestep < 1000:
        traci.simulationStep()
        pedestrians = traci.person.getIDList()
        vehicles = traci.vehicle.getIDList()
        for ped in pedestrians:
            pos_ped = traci.person.getPosition(ped)
            row_ped.append({'data_timestep': int(data_timestep), 'ped_id': str(ped), 'ped_x': float(pos_ped[0]),
                            'ped_y': float(pos_ped[1])})
        for veh in vehicles:
            pos_veh = traci.vehicle.getPosition(veh)
            row_veh.append({'data_timestep': int(data_timestep), 'vehicle_id': str(veh), 'vehicle_x': float(pos_veh[0]),
                            'vehicle_y': float(pos_veh[1])})
        data_timestep += 1

    # stop SUMO
    traci.close()
    ped = pd.DataFrame(data=row_ped, columns=columns_ped)
    veh = pd.DataFrame(data=row_veh, columns=columns_veh)
    ped.to_csv('ped.csv', index=False)
    veh.to_csv('veh.csv', index=False)