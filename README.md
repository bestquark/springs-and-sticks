# Learning with springs and sticks 

Here you can find the code for the project "Learning with springs and sticks". `springsticks.ipynb` contains the code for the figures of this project. To run it, first install the required packages by running `pip install -r requirements.txt`. 

The main code can be found in the `src/` folder. Animations and figures for this project can be found in the `anis/` and `figs/` folders, respectively.

To classify the MNIST dataset with the springstick model, run the `mnist.py` script.


For profiling 

```
!gprof2dot -f pstats profile_stats.prof | dot -Tpng -o profile.png
```