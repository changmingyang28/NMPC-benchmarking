import os
import signal
import sys
import time

import example_robot_data
import numpy as np

import crocoddyl
from crocoddyl.utils.pendulum import (
    ActuationModelDoublePendulum,
    CostModelDoublePendulum,
)

WITHDISPLAY = "display" in sys.argv or "CROCODDYL_DISPLAY" in os.environ
WITHPLOT = "plot" in sys.argv or "CROCODDYL_PLOT" in os.environ
signal.signal(signal.SIGINT, signal.SIG_DFL)

# Loading the double pendulum model
panda = example_robot_data.load("panda")
model = panda.model
print(model)

