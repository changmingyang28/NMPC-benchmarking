import os
import sys
import numpy as np
import crocoddyl
import pinocchio
from sys import argv
from os.path import dirname, join, abspath
import math
import signal
import robotoc
import time

WITHDISPLAY = "display" in sys.argv or "CROCODDYL_DISPLAY" in os.environ
WITHPLOT = "plot" in sys.argv or "CROCODDYL_PLOT" in os.environ
signal.signal(signal.SIGINT, signal.SIG_DFL)

pinocchio_model_dir = join(dirname(dirname(str(abspath(__file__)))), "robotoc")
urdf_filename = pinocchio_model_dir + '/iiwa_description/urdf/iiwa14.urdf' if len(argv)<2 else argv[1]
robot_model = pinocchio.buildModelFromUrdf(urdf_filename)
model_info = robotoc.RobotModelInfo()

model_info.urdf_path = "robotoc/iiwa_description/urdf/iiwa14.urdf"
robot = robotoc.Robot(model_info)
# robot.set_joint_effort_limit(np.full(robot.dimu(), 50))
# robot.set_joint_velocity_limit(np.full(robot.dimv(), 0.5 * math.pi))

#print(robot)
#print(robot_model)

q0 = np.array([0, 0.5 * math.pi, 0, 0.5 * math.pi, 0, 0.5 * math.pi, 0])
x0 = np.concatenate([q0, pinocchio.utils.zero(robot_model.nv)])
state = crocoddyl.StateMultibody(robot_model)
actuation = crocoddyl.ActuationModelFull(state)
nu = state.nv
runningCostModel = crocoddyl.CostModelSum(state)
terminalCostModel = crocoddyl.CostModelSum(state)


framePlacementResidual = crocoddyl.ResidualModelFramePlacement(
    state,
    robot_model.getFrameId("iiwa_link_ee"),
    pinocchio.SE3(np.eye(3), np.array([0.6, 0.2, 0.5])),
    nu,
)
uResidual = crocoddyl.ResidualModelControl(state, nu)
xResidual = crocoddyl.ResidualModelState(state, x0, nu)
goalTrackingCost = crocoddyl.CostModelResidual(state, framePlacementResidual)
xRegCost = crocoddyl.CostModelResidual(state, xResidual)
uRegCost = crocoddyl.CostModelResidual(state, uResidual)

# Then let's added the running and terminal cost functions
runningCostModel.addCost("gripperPose", goalTrackingCost, 1)
runningCostModel.addCost("xReg", xRegCost, 1e-1)
runningCostModel.addCost("uReg", uRegCost, 1e-1)
terminalCostModel.addCost("gripperPose", goalTrackingCost, 1e3)

# Next, we need to create an action model for running and terminal knots. The
# forward dynamics (computed using ABA) are implemented
# inside DifferentialActionModelFreeFwdDynamics.
dt = 1e-2
runningModel = crocoddyl.IntegratedActionModelEuler(
    crocoddyl.DifferentialActionModelFreeFwdDynamics(
        state, actuation, runningCostModel
    ),
    dt,
)
terminalModel = crocoddyl.IntegratedActionModelEuler(
    crocoddyl.DifferentialActionModelFreeFwdDynamics(
        state, actuation, terminalCostModel
    ),
    0.0,
)

# For this optimal control problem, we define 100 knots (or running action
# models) plus a terminal knot
T = 100
problem = crocoddyl.ShootingProblem(x0, [runningModel] * T, terminalModel)

# Creating the DDP solver for this OC problem, defining a logger
solver = crocoddyl.SolverBoxFDDP(problem)
cameraTF = [2.0, 2.68, 0.54, 0.2, 0.62, 0.72, 0.22]
if WITHDISPLAY:
    try:
        import gepetto

        gepetto.corbaserver.Client()
        display = crocoddyl.GepettoDisplay(robot_model, 4, 4, cameraTF, floor=False)
        if WITHPLOT:
            solver.setCallbacks(
                [
                    crocoddyl.CallbackVerbose(),
                    crocoddyl.CallbackLogger(),
                    crocoddyl.CallbackDisplay(display),
                ]
            )
        else:
            solver.setCallbacks(
                [crocoddyl.CallbackVerbose(), crocoddyl.CallbackDisplay(display)]
            )
    except Exception:
        display = crocoddyl.MeshcatDisplay(robot_model)
if WITHPLOT:
    solver.setCallbacks(
        [
            crocoddyl.CallbackVerbose(),
            crocoddyl.CallbackLogger(),
        ]
    )
else:
    solver.setCallbacks([crocoddyl.CallbackVerbose()])
solver.getCallbacks()[0].precision = 3
solver.getCallbacks()[0].level = crocoddyl.VerboseLevel._2

# Solving it with the solver algorithm
solver.solve()

print(
    "Finally reached = ",
    solver.problem.terminalData.differential.multibody.pinocchio.oMf[
        robot_model.getFrameId("iiwa_link_ee")
    ].translation.T,
)

# Plotting the solution and the solver convergence
if WITHPLOT:
    log = solver.getCallbacks()[1]
    crocoddyl.plotOCSolution(log.xs, log.us, figIndex=1, show=False)
    crocoddyl.plotConvergence(
        log.costs, log.u_regs, log.x_regs, log.grads, log.stops, log.steps, figIndex=2
    )

# Visualizing the solution in gepetto-viewer
if WITHDISPLAY:
    display.rate = -1
    display.freq = 1
    while True:
        display.displayFromSolver(solver)
        time.sleep(1.0)


