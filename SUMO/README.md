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

The old part uses [traci](https://pypi.org/project/traci/) package. We can just run [sumo.py](./new/sumo.py) to get [veh.csv](./new/veh.csv) and [ped.csv](./new/ped.csv).