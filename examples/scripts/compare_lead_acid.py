#
# Compare lead-acid battery models
#
import argparse
import numpy as np
import pybamm

parser = argparse.ArgumentParser()
parser.add_argument(
    "--debug", action="store_true", help="Set logging level to 'DEBUG'."
)
args = parser.parse_args()
if args.debug:
    pybamm.set_logging_level("DEBUG")
else:
    pybamm.set_logging_level("INFO")

# load models
models = [
    pybamm.lead_acid.NewmanTiedemann(),
    pybamm.lead_acid.LOQS(),
    pybamm.lead_acid.FOQS(),
    pybamm.lead_acid.FOQSAverageCorrection(),
    pybamm.lead_acid.CompositeExtended(),
    # pybamm.lead_acid.Composite(
    #     {"surface form": "differential", "side reactions": ["oxygen"]}
    # ),
    # # pybamm.lead_acid.CompositeExtended(
    # #     {"surface form": "differential", "side reactions": ["oxygen"]}
    # # ),
    # # pybamm.lead_acid.CompositeAverageCorrection(
    # #     {"surface form": "differential", "side reactions": ["oxygen"]}
    # # ),
    # pybamm.lead_acid.LOQS(
    #     {"surface form": "differential", "side reactions": ["oxygen"]}
    # ),
    # # pybamm.lead_acid.FOQS({"surface form": "algebraic"}),
]

# load parameter values and process models and geometry
param = models[0].default_parameter_values
param.update({"Typical current [A]": 20, "Initial State of Charge": 1})
for model in models:
    param.process_model(model)

# discretise models
var = pybamm.standard_spatial_vars
var_pts = {var.x_n: 25, var.x_s: 41, var.x_p: 34, var.y: 10, var.z: 10}
for model in models:
    geometry = model.default_geometry
    param.process_geometry(geometry)
    mesh = pybamm.Mesh(geometry, model.default_submesh_types, var_pts)
    disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
    disc.process_model(model)

# solve model
solutions = [None] * len(models)
t_eval = np.linspace(0, 5, 100)
for i, model in enumerate(models):
    solution = model.default_solver.solve(model, t_eval)
    solutions[i] = solution

# plot
output_variables = [
    "Electrolyte potential [V]",
    "Electrolyte concentration [Molar]",
    "X-averaged electrolyte concentration [Molar]",
    "Transverse volume-averaged acceleration [m.s-2]",
    "Terminal voltage [V]",
]
plot = pybamm.QuickPlot(models, mesh, solutions, output_variables)
plot.dynamic_plot()
