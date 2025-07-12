import os
import sys
import numpy as np
import pathlib
import crocoddyl
import pinocchio
import time

# Get the path to the urdf
from example_robot_data.path import EXAMPLE_ROBOT_DATA_MODEL_DIR
from crocoddyl.utils.pendulum import (
    ActuationModelDoublePendulum,
    CostModelDoublePendulum,
)

urdf_model_path = pathlib.Path(
    "double_pendulum_description", "urdf", "double_pendulum_simple.urdf"
)
urdf_model_path = os.path.join(EXAMPLE_ROBOT_DATA_MODEL_DIR, urdf_model_path)

# Now load the model (using pinocchio)
robot = pinocchio.robot_wrapper.RobotWrapper.BuildFromURDF(str(urdf_model_path))

# The model loaded from urdf (via pinicchio)
print(robot.model)
 # reduced artificially the torque limits

# Create a multibody state from the pinocchio model.
state = crocoddyl.StateMultibody(robot.model)

class AcrobotActuationModel(crocoddyl.ActuationModelAbstract):
    def __init__(self, state):
        nu = 2  # Control dimension
        crocoddyl.ActuationModelAbstract.__init__(self, state, nu=nu)

    def calc(self, data, x, u):
        assert len(data.tau) == 2
        # Map the control dimensions to the joint torque
        data.tau[0] = u[0]
        data.tau[1] = u[1]

    def calcDiff(self, data, x, u):
        # Specify the actuation jacobian
        data.dtau_du[0,0] = 1
        data.dtau_du[0,1] = 0
        data.dtau_du[1,0] = 0
        data.dtau_du[1,1] = 1

# Also see ActuationModelFloatingBase and ActuationModelFull
actuationModel = AcrobotActuationModel(state)

dt = 2e-3  # Time step
T = 500
runningCostModel = crocoddyl.CostModelSum(state, nu=2)
terminalCostModel = crocoddyl.CostModelSum(state, nu=2)

xref = np.array([0, 0, 0, 0])  # Desired state
stateResidual = crocoddyl.ResidualModelState(state, xref=xref, nu=2)
stateCostModel = crocoddyl.CostModelResidual(state, stateResidual)
runningCostModel.addCost("state_cost", cost=stateCostModel, weight=0.01)   #1e-5/dt
terminalCostModel.addCost("state_cost", cost=stateCostModel, weight=100)

controlResidual = crocoddyl.ResidualModelControl(state, nu=2)

bounds = crocoddyl.ActivationBounds(np.array([-1.0, -1.0]), np.array([1.0, 1.0]))
activation = crocoddyl.ActivationModelQuadraticBarrier(bounds)
controlCost = crocoddyl.CostModelResidual(
    state, activation=activation, residual=controlResidual
)
runningCostModel.addCost("control_cost", cost=controlCost, weight=0.01)   #1e-1

runningModel = crocoddyl.IntegratedActionModelEuler(
    crocoddyl.DifferentialActionModelFreeFwdDynamics(
        state, actuationModel, runningCostModel
    ),
    dt,
)
terminalModel = crocoddyl.IntegratedActionModelEuler(
    crocoddyl.DifferentialActionModelFreeFwdDynamics(
        state, actuationModel, terminalCostModel
    ),
    0.0,
)

q0 = np.zeros((state.nq,))  # Inital joint configurations
q0[0] = -np.pi   # Down
v0 = np.zeros((state.nv,))  # Initial joint velocities
x0 = np.concatenate((q0, v0))  # Inital robot state
problem = crocoddyl.ShootingProblem(x0, [runningModel] * T, terminalModel)

#solver type:  DDP  FDDP  BoxDDP BoxFDDP Intro
solver = crocoddyl.SolverFDDP(problem)    

# Solve
callbacks = []
callbacks.append(crocoddyl.CallbackLogger())
callbacks.append(crocoddyl.CallbackVerbose())
solver.setCallbacks(callbacks)

start_time= time.time()
solver.solve([], [], 300, False, 1e-5)
end_time  = time.time()
print((end_time-start_time)*1000,"ms")

log = solver.getCallbacks()[0]

import matplotlib.pyplot as plt

crocoddyl.plotOCSolution(
    xs=log.xs, us=log.us, show=False, figIndex=1, figTitle="Solution"
)
fig = plt.gcf()
axs = fig.axes
for ax in axs:
    ax.grid(True)

crocoddyl.plotConvergence(
    log.costs,
    log.u_regs,
    log.x_regs,
    log.grads,
    log.stops,
    log.steps,
    show=False,
    figIndex=2,
)
fig = plt.gcf()
axs = fig.axes
for ax in axs:
    ax.grid(True)

#plt.show()
plt.savefig("result.png")