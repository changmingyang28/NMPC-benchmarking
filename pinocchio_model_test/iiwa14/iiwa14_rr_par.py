import robotoc
import numpy as np
import math
import matplotlib.pyplot as plt

model_info = robotoc.RobotModelInfo()
model_info.urdf_path = "robotoc/iiwa_description/urdf/iiwa14.urdf"
robot = robotoc.Robot(model_info)

# Change the limits from the default parameters.
robot.set_joint_effort_limit(np.full(robot.dimu(), 5))

# Create a cost function.
cost = robotoc.CostFunction()
config_cost = robotoc.ConfigurationSpaceCost(robot)
q_ref = np.array([0, 0.5*math.pi, 0, 0.5*math.pi, 0, 0.5*math.pi, 0]) 

config_cost.set_q_ref(q_ref)
config_cost.set_q_weight(np.full(robot.dimv(), 10))
config_cost.set_q_weight_terminal(np.full(robot.dimv(), 10))
config_cost.set_v_weight(np.full(robot.dimv(), 0.01))
config_cost.set_v_weight_terminal(np.full(robot.dimv(), 0.01))
config_cost.set_a_weight(np.full(robot.dimv(), 0.001))
#config_cost.set_u_weight(np.full(robot.dimv(), 0.1))
cost.add("config_cost", config_cost)

# Create joint constraints.
constraints = robotoc.Constraints(barrier_param=1.0e-03, fraction_to_boundary_rule=0.995)
# joint_torques_lower   = robotoc.JointTorquesLowerLimit(robot)
# joint_torques_upper   = robotoc.JointTorquesUpperLimit(robot)
# constraints.add("joint_torques_lower", joint_torques_lower)
# constraints.add("joint_torques_upper", joint_torques_upper)

# Create the OCP solver for unconstrained rigid-body systems.
T = 1.0
N = 100
ocp = robotoc.OCP(robot=robot, cost=cost, constraints=constraints, T=T, N=N)
solver_options = robotoc.SolverOptions()
solver_options.nthreads = 4
solver_options.enable_benchmark = True
ocp_solver = robotoc.UnconstrOCPSolver(ocp=ocp, solver_options=solver_options)

solver_options.max_iter = 1000

# Initial time and intial state 
t = 0.0
q = np.array([0.5*math.pi, 0, 0.5*math.pi, 0, 0.5*math.pi, 0, 0.5*math.pi]) 
v = np.zeros(robot.dimv())

print("----- Solves the OCP by Riccati recursion algorithm. -----")
ocp_solver.discretize(t)
ocp_solver.set_solution("q", q)
ocp_solver.set_solution("v", v)
ocp_solver.init_constraints()
print("Initial KKT error: ", ocp_solver.KKT_error(t, q, v))
ocp_solver.solve(t, q, v, init_solver=True)
print("KKT error after convergence: ", ocp_solver.KKT_error(t, q, v))
print(ocp_solver.get_solver_statistics())

# Solves the OCP by ParNMPC algorithm.
solver_options.nthreads = 8
parnmpc_solver = robotoc.UnconstrParNMPCSolver(ocp=ocp, solver_options=solver_options)

print("\n----- Solves the OCP by ParNMPC algorithm. -----")
parnmpc_solver.discretize(t)
parnmpc_solver.set_solution("q", q)
parnmpc_solver.set_solution("v", v)
parnmpc_solver.init_constraints()
parnmpc_solver.init_backward_correction()
print("Initial KKT error: ", parnmpc_solver.KKT_error(t, q, v))
parnmpc_solver.solve(t, q, v, init_solver=True)
print("KKT error after convergence: ", parnmpc_solver.KKT_error(t, q, v))
print(parnmpc_solver.get_solver_statistics())
print(parnmpc_solver.get_solver_statistics().cpu_time)
rrtime=ocp_solver.get_solver_statistics().cpu_time
partime=parnmpc_solver.get_solver_statistics().cpu_time

rrq=ocp_solver.get_solution('q')
rru=ocp_solver.get_solution('u')
parq=parnmpc_solver.get_solution('q')
paru=parnmpc_solver.get_solution('u')

rrq_s= [[],[],[],[],[],[],[]]     
rru_s=[[],[],[],[],[],[],[]]                
parq_s=[]
paru_s=[]

for i in range(0,N):
    a=rrq[i]
    b=rru[i]
    c=parq[i]
    d=paru[i]
    parq_s.append(c[0])   
    paru_s.append(d[0])
    for j in range(0,7):
        rrq_s[j].append(a[j])
        rru_s[j].append(b[j])

plt.figure()
plt.plot(rrq_s[0],color='r')
plt.plot(rrq_s[1],color='b')
plt.plot(rrq_s[2],color='c')
plt.plot(rrq_s[3],color='k')
plt.plot(rrq_s[4],color='m')
plt.plot(rrq_s[5],color='y')
plt.plot(rrq_s[6],color='g')
plt.xlabel('Knots n',fontsize=15)
plt.ylabel('Position of Joints',fontsize=15)
plt.savefig("test2.png")

plt.figure()
plt.plot(parq_s,color='b')
plt.savefig("test3.png")

