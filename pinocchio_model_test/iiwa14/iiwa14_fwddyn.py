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
import iiwa14_rr_par as rrpar



pinocchio_model_dir = join(dirname(dirname(str(abspath(__file__)))))
urdf_filename = pinocchio_model_dir + '/iiwa14.urdf' if len(argv)<2 else argv[1]
robot= pinocchio.buildModelFromUrdf(urdf_filename)

state = crocoddyl.StateMultibody(robot)
print(robot)


# Define the control signal to actuated joint mapping
class AcrobotActuationModel(crocoddyl.ActuationModelAbstract):
    def __init__(self, state):
        nu = 7  # Control dimension
        crocoddyl.ActuationModelAbstract.__init__(self, state, nu=nu)

    def calc(self, data, x, u):
        assert len(data.tau) == 7
        # Map the control dimensions to the joint torque
        data.tau[0] = u[0]
        data.tau[1] = u[1]
        data.tau[2] = u[2]
        data.tau[3] = u[3]
        data.tau[4] = u[4]
        data.tau[5] = u[5]
        data.tau[6] = u[6]

    def calcDiff(self, data, x, u):
        # Specify the actuation jacobian
        data.dtau_du = np.identity(7)
# Also see ActuationModelFloatingBase and ActuationModelFull
actuationModel = AcrobotActuationModel(state)

dt = 1e-2  # Time step
T = 100  # Number of knots

# Cost models
runningCostModel = crocoddyl.CostModelSum(state, nu=actuationModel.nu)
terminalCostModel = crocoddyl.CostModelSum(state, nu=actuationModel.nu)

# Add a cost for the configuration positions and velocities
xref = np.array([0, 0.5*math.pi, 0, 0.5*math.pi, 0, 0.5*math.pi, 0, 0, 0, 0, 0, 0, 0, 0])  # Desired state
stateResidual = crocoddyl.ResidualModelState(state, xref=xref, nu=actuationModel.nu)
stateCostModel = crocoddyl.CostModelResidual(state, stateResidual)
runningCostModel.addCost("state_cost", cost=stateCostModel, weight=1)   #1e-5/dt
terminalCostModel.addCost("state_cost", cost=stateCostModel, weight=100)    #1000   

# Add a cost on control
controlResidual = crocoddyl.ResidualModelControl(state, nu=actuationModel.nu)
bounds = crocoddyl.ActivationBounds(np.array([-50.0,-50.0,-50.0,-50.0,-50.0,-50.0,-50.0]),np.array([50.0,50.0,50.0,50.0,50.0,50.0,50.0]))
activation = crocoddyl.ActivationModelQuadraticBarrier(bounds)
controlCost = crocoddyl.CostModelResidual(
    state, activation=activation, residual=controlResidual
)
runningCostModel.addCost("control_cost", cost=controlCost, weight=1e-1 / dt)   #1e-1

# Create the action models for the state
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

# Define a shooting problem
# Inital joint configurations
q0 = np.array([0.5*math.pi, 0, 0.5*math.pi, 0, 0.5*math.pi, 0, 0.5*math.pi])   # Down
v0 = np.zeros((state.nv,))  # Initial joint velocities
x0 = np.concatenate((q0, v0))  # Inital robot state
problem = crocoddyl.ShootingProblem(x0, [runningModel] * T, terminalModel)

# Now stabilize the acrobot using FDDP
solver = crocoddyl.SolverFDDP(problem)

# Solve
callbacks = []
callbacks.append(crocoddyl.CallbackLogger())
callbacks.append(crocoddyl.CallbackVerbose())
solver.setCallbacks(callbacks)

FDDP_time=[]

start_time= time.time()
solver.solve([], [], 300, False, 1e-5)
end_time  = time.time()
fddptime=(end_time-start_time)*1000

    
#
fddp_log = solver.getCallbacks()[0]

#solve with invDDP
runningModel = crocoddyl.IntegratedActionModelEuler(
    crocoddyl.DifferentialActionModelFreeInvDynamics(
        state, actuationModel, runningCostModel
    ),
    dt,
)
terminalModel = crocoddyl.IntegratedActionModelEuler(
    crocoddyl.DifferentialActionModelFreeInvDynamics(
        state, actuationModel, terminalCostModel
    ),
    dt,
)
 
#problem = crocoddyl.ShootingProblem(x0, [runningModel] * T, terminalModel)
solver = crocoddyl.SolverIntro(problem)

callbacks = []
callbacks.append(crocoddyl.CallbackLogger())
callbacks.append(crocoddyl.CallbackVerbose())
solver.setCallbacks(callbacks)

start_time= time.time()
solver.solve()
end_time  = time.time()
Iddptime=(end_time-start_time)*1000
print("InvDDP_time:",Iddptime,"ms")
iDDP_log = solver.getCallbacks()[0]




# Plotting the solution and the DDP convergence

import iiwa14_rr_par
import matplotlib.pyplot as plt


fwdDDP_x0=[]
fwdDDP_u0=[]
invDDP_x0=[]
invDDP_u0=[]
for i in range(0,T):
    fddpx=fddp_log.xs[i]
    iddpx=iDDP_log.xs[i]
    fwdDDP_x0.append(fddpx[0])
    invDDP_x0.append(iddpx[0])
    fddpu=fddp_log.us[i]
    iddpu=iDDP_log.us[i]
    fwdDDP_u0.append(fddpu[0])
    invDDP_u0.append(iddpu[0])

plt.figure()
plt.plot(fwdDDP_x0,color='b',label='FwdDDP')
plt.plot(invDDP_x0,color='r',label='InvDDP')
plt.plot(rrpar.rrq_s[0],color='g',label='Recatti Recursion')
plt.plot(rrpar.parq_s,color='c',label='ParNMPC')
plt.legend(fontsize=12)
plt.savefig("iiwa_fwd_x0.png")

plt.figure()
plt.plot(fwdDDP_u0,color='b',label='FwdDDP')
plt.plot(invDDP_u0,color='r',label='InvDDP')
plt.plot(rrpar.rru_s[0],color='g',label='Recatti Recursion')
plt.plot(rrpar.paru_s,color='c',label='ParNMPC')
plt.legend(fontsize=12)
plt.savefig("iiwa_fwd_u0.png")


materials=['Riccati Recursion','ParNMPC','FwdDDP','InvDDP']
meandata=[iiwa14_rr_par.rrtime,iiwa14_rr_par.partime,fddptime,Iddptime]
x_pos=np.arange(4)
fig,ax= plt.subplots()
ax.bar(x_pos,meandata,align='center',alpha=0.5,ecolor='black',capsize=10)
ax.set_xticks(x_pos)
ax.set_xticklabels(materials,fontsize=12)
ax.yaxis.grid(True)
plt.tight_layout()
plt.legend(fontsize=10)
plt.savefig('iiwa14 CPU Time.png')
