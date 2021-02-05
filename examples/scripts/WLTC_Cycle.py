import pybamm
import numpy as np
import pandas as pd

pybamm.set_logging_level("INFO")

# load models
model = pybamm.lithium_ion.SPM({"operating mode": "power"})

# load parameter values and process models and geometry
param = model.default_parameter_values

drive_cycle = pd.read_csv("../../pybamm/input/drive_cycles/WLTC.csv", comment="#", header=None).to_numpy()

""" Scale down down the power to cell level
Note that the supplied profile represents the power absorption 
of a vehicle with a power of 85 kW and mass 1500 kg on the WLTC cycle.
,
1) Convert kW to W (multiply by 1000)
2) Hypotesis on the battery package: 
18650 cell with 3 Ah capacity
100 cells in series x 20 parallels: 2000 cells (divide by 2000)
"""

conv=1000 # kW to W
series=100
parallel=20

drive_cycle[:, 1] = conv*drive_cycle[:, 1]/(series*parallel) # Power on the cell

# Create interpolant
timescale = param.evaluate(model.timescale)
power_interpolant = pybamm.Interpolant(drive_cycle, timescale * pybamm.t)

# discharge
param.update({"Power function [W]": power_interpolant}, check_already_exists=False)

sim = pybamm.Simulation(model, parameter_values=param)
t_eval = np.linspace(0, 1800, 18001)
sim.solve(t_eval)


sim.plot(["Current [A]", "Terminal voltage [V]"], time_unit='seconds', variable_limits='tight')

