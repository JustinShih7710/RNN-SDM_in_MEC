# SUMO part

We use SUMO to generate our dataset and simulation data.

## Introduction

This directory contain two directory, [new](./new) and [old](./old).

### The old part step

The old part uses sumo.exe to generate all simulation data.First step, we need to change [osm.sumocfg](./old/osm.sumocfg). Second, we need to open this with sumo. You can also use command line:

```
run.bat
```

After we have [fulloutput.xml](./old/fulloutput.xml). We need to use the tools SUMO provided and named [xml2csv.py](./old/xml2csv.py), you need to run it with command line:

```
python xml2csv.py fulloutput.xml -o fulloutput.csv
```

Finally, we can run [readcsv.py](./old/readcsv.py) to get [veh.csv](./old/veh.csv).

### The new part step

The new part uses [traci](https://pypi.org/project/traci/) package.After SUMO generate [osm.sumocfg](./new/osm.sumocfg), we can just run [sumo.py](./new/sumo.py) to get [veh.csv](./new/veh.csv) and [ped.csv](./new/ped.csv).

If the SUMO didn't generate route file, you can run the script below in command line,

For vehicle routes:
```
python "$SUMO_HOME/tools/randomTrips.py" -n <your net file> -o <your vehicle route file> --begin 0 --end 3600 --period 2
```
For pedestrian routes:

```
python "$SUMO_HOME/tools/randomTrips.py" -n <your net file> -o <your pedestrian route file> --begin 0 --end 3600 --period 2 --pedestrians
```
For more information, refer to the [randomTrips.py](https://sumo.dlr.de/docs/Tools/Trip.html) documentation.