# Learning with springs and sticks 

[![Website](https://img.shields.io/badge/Website-Live-blue)](https://bestquark.github.io/springs-and-sticks/)

Here you can find the code for the project "Learning with springs and sticks". This project introduces an ML model based on a dissipative mechanical system composed of springs and sticks as shown below:

<p style="text-align:center;">
<img src="anis/animation_paraboloid.gif" alt="Springs and sticks dynamics" width="40%">
</p>

The main code can be found in the `src/` folder and the code for generating the paper's figures can be found in `paper_imgs/`. Animations and figures for this project can be found in the `anis/` and `figs/` folders, respectively.

<!-- A simple tutorial on the model is provided in the `springs_and_sticks.ipynb` file. This notebook contains minimal working examples of the model. -->

### Installation

To use this model, clone this repository, create a virtual environment and install this package by running

```
python -m venv springsenv
source springsenv/bin/activate
git clone https://github.com/bestquark/springs-and-sticks.git
cd springs-and-sticks
pip install -e .
```

### Usage

After installing the package, you can run the model as:

```python
import torch
import torchsde

from ss.model import GS3DE

torch.manual_seed(0)
# dataset
u_i = torch.linspace(0, 2*torch.pi, 20).unsqueeze(1)
y_i = 0.1*torch.randn_like(u_i) + 1 + (-1/3)*u_i

# boundaries of sticks grid
boundaries = (torch.min(u_i, dim=0).values, torch.max(u_i, dim=0).values)
n_labels = y_i.shape[1]

# number of simulations and time steps
ns = 1 
batch_size, t_size = 100, 1000
sde = GS3DE(ns,boundaries, n_labels, friction=4, temp=1e-1, k=1, M=1)

# time integral of the equations of motion
ts = torch.linspace(0, 5, t_size)
theta0 = (torch.rand(size=(batch_size, sde.state_size)))
with torch.no_grad():
    sde.update_data(u_i, y_i)
    thetas = torchsde.sdeint(sde, theta0, ts, method='euler') 
```

### Development

If you are profiling, run the following to visualize the results:

```
!gprof2dot -f pstats profile_stats.prof | dot -Tpng -o profile.png
```
