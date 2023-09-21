# RNN part

After we have DataSet from Sumo, we start train our model.

You can look at methods such as `SimpleRNN()`, `Xavier_RNN()`, `multi_layer_SimpleRNN()`, etc. to build our RNN model.

We test hyperparameter and different layer many times, we finally chose `Xavier_RNN()` which can predict location xy among 50m.