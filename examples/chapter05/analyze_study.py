import os
import sys

import optuna

db_path = sys.argv[1]

study_dir = os.path.dirname(db_path)
study_name = os.path.split(study_dir)[1]
study = optuna.load_study(storage=f"sqlite:///{db_path}", study_name=study_name)

fig = optuna.visualization.plot_param_importances(study)
fig.write_image("importances.pdf")

fig = optuna.visualization.plot_optimization_history(study)
fig.write_image("history.pdf")

fig = optuna.visualization.plot_timeline(study)
fig.write_image("timeline.pdf")
