# Simu5G part

After we train our model, we need to build a 5G simulation environment to evaluate our proposed method.

The main function is [simu5G.py](simu5G.py).

[MEC_server.py](MEC_server.py) is used to generate mec nodes, and it output is [MEC_server.json](MEC_server.json)

[UE.py](UE.py) and [MEC.py](MEC.py) is a class to create object while simulation.

[timeslot_mec_content.py](timeslot_mec_content.py) and [Result.py](Result.py) is used to evaluate our 5G hyperparameter is useful.