import os
import sys
import numpy as np
import pathlib
import crocoddyl
import pinocchio
import time
import matplotlib.pyplot as plt
import dp_robotoc

# Get the path to the urdf
from example_robot_data.path import EXAMPLE_ROBOT_DATA_MODEL_DIR

urdf_model_path = pathlib.Path(
    "double_pendulum_description", "urdf", "double_pendulum_simple.urdf"
)
urdf_model_path = os.path.join(EXAMPLE_ROBOT_DATA_MODEL_DIR, urdf_model_path)

# Now load the model (using pinocchio)
robot = pinocchio.robot_wrapper.RobotWrapper.BuildFromURDF(str(urdf_model_path))

# The model loaded from urdf (via pinicchio)
#print(robot.model)
# reduced artificially the torque limits

# Create a multibody state from the pinocchio model.
state = crocoddyl.StateMultibody(robot.model)

# Define the control signal to actuated joint mapping
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

dt = 3e-2 # Time step
T = 100 # Number of knots

# Cost models
runningCostModel = crocoddyl.CostModelSum(state, nu=actuationModel.nu)
terminalCostModel = crocoddyl.CostModelSum(state, nu=actuationModel.nu)

# Add a cost for the configuration positions and velocities
xref = np.array([0, 0, 0, 0])  # Desired state
stateResidual = crocoddyl.ResidualModelState(state, xref=xref, nu=actuationModel.nu)
stateCostModel = crocoddyl.CostModelResidual(state, stateResidual)
runningCostModel.addCost("state_cost", cost=stateCostModel, weight=1e-2 / dt)   
terminalCostModel.addCost("state_cost", cost=stateCostModel, weight=100)       

# Add a cost on control
controlResidual = crocoddyl.ResidualModelControl(state, nu=actuationModel.nu)
bounds = crocoddyl.ActivationBounds(np.array([-1, -1]), np.array([1, 1]))
activation = crocoddyl.ActivationModelQuadraticBarrier(bounds)
controlCost = crocoddyl.CostModelResidual(
    state, activation=activation, residual=controlResidual
)
runningCostModel.addCost("control_cost", cost=controlCost, weight=0.1 / dt)

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
    0.0,)

# Define a shooting problem
q0 = np.zeros((state.nq,))  # Inital joint configurations
q0[0] = -np.pi # Down
v0 = np.zeros((state.nv,))  # Initial joint velocities
x0 = np.concatenate((q0, v0))  # Inital robot state
problem = crocoddyl.ShootingProblem(x0, [runningModel] * T, terminalModel)

# Solve with ForwardDDP
solver = crocoddyl.SolverFDDP(problem)
callbacks = []
callbacks.append(crocoddyl.CallbackLogger())
callbacks.append(crocoddyl.CallbackVerbose())
solver.setCallbacks(callbacks)

FDDP_time=[]
for i in range(0,10):
    start_time= time.time()
    solver.solve()
    end_time  = time.time()
    ftime= (end_time-start_time)*1000
    FDDP_time.append(ftime)

    FwdDDP_log = solver.getCallbacks()[0]


# Solve with inversDDP
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

IDDP_time=[]
for i in range (0,2):
    start_time= time.time()
    solver.solve()
    end_time  = time.time()
    itime= (end_time-start_time)*1000
    IDDP_time.append(itime)

  
InvDDP_log = solver.getCallbacks()[0]

FwdDDP_xs=[[],[],[],[]]
InvDDP_xs=[[],[],[],[]]
FwdDDP_us=[[],[]]
InvDDP_us=[[],[]]
for i in range (0,T):
    fddpx=FwdDDP_log.xs[i]
    FwdDDP_xs[0].append(fddpx[0])
    iddpx=InvDDP_log.xs[i]
    InvDDP_xs[0].append(iddpx[0])
    
    fddpu=FwdDDP_log.us[i]
    FwdDDP_us[0].append(fddpu[0])
    iddpu=InvDDP_log.us[i]
    InvDDP_us[0].append(iddpu[0])
    
q_s = dp_robotoc.q_s
u_s = dp_robotoc.u_s

plt.figure()
plt.plot(FwdDDP_xs[0],color='b',label='FwdDDP')
plt.plot(InvDDP_xs[0],color='r',label='InvDDP')
plt.plot(q_s[0],color='g',label='RicattiRecursion')
plt.plot(q_s[2],color='darkorange',label='ParNMPC')
plt.legend(fontsize=12)
plt.savefig("dp_x0.png")

plt.figure()
plt.plot(FwdDDP_us[0],color='b',label='FwdDDP')
plt.plot(InvDDP_us[0],color='r',label='InvDDP')
plt.plot(u_s[0],color='g',label='RicattiRecursion')
plt.plot(u_s[2],color='darkorange',label='ParNMPC')
plt.legend(fontsize=12)
plt.savefig("dp_u0.png")        

ftime_mean=np.mean(FDDP_time)
itime_mean=np.mean(IDDP_time)
ftime_std =np.std(FDDP_time)
itime_std =np.std(IDDP_time)

meandata=[dp_robotoc.ocpcputime_mean, dp_robotoc.parcputime_mean, ftime_mean, itime_mean]
stddata =[dp_robotoc.ocpcputime_std , dp_robotoc.parcputime_std/3,  ftime_std,  itime_std ]
materials=['Riccati Recursion','ParNMPC','FwdDDP','InvDDP']

print(meandata,stddata)

x_pos=np.arange(4)

fig,ax= plt.subplots()
ax.bar(x_pos,meandata,yerr=stddata,align='center',alpha=0.5,ecolor='black',capsize=10)
# for a,b in zip(materials,meandata):
#     ax.text(a,b,'%.2f'%b,ha='center',va='bottom',fontsize=12)
ax.set_xticks(x_pos)
ax.set_xticklabels(materials,fontsize=12)
ax.yaxis.grid(True)

plt.tight_layout()
plt.legend
plt.savefig('dp CPU Time.png')












    






    


    
