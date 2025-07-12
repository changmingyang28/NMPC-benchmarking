import robotoc     
import numpy as np
import math
import robotoc
from robotoc import PlotConvergence
from robotoc import solver
import matplotlib.pyplot as plt

model_info = robotoc.RobotModelInfo()
model_info.urdf_path = "robotoc/iiwa_description/urdf/double_pendulum_simple.urdf"
robot = robotoc.Robot(model_info)

robot.set_joint_effort_limit(np.full(robot.dimu(), 1))
# robot.set_joint_velocity_limit(np.full(robot.dimv(), 5))

#print(robot)

cost = robotoc.CostFunction()
config_cost = robotoc.ConfigurationSpaceCost(robot)
q_ref = np.array([0,0])
config_cost.set_q_ref(q_ref)
config_cost.set_q_weight(np.full(robot.dimv(), 1))
config_cost.set_v_weight(np.full(robot.dimv(), 1))
config_cost.set_q_weight_terminal(np.full(robot.dimv(), 100))
config_cost.set_v_weight_terminal(np.full(robot.dimv(), 100))
#config_cost.set_a_weight(np.full(robot.dimv(), 0.01))
config_cost.set_u_weight(np.full(robot.dimv(), 0.1))
cost.add("config_cost", config_cost)

# Create joint constraints.
constraints = robotoc.Constraints(
    barrier_param=1.0e-03, fraction_to_boundary_rule=0.995
    )
joint_torques_lower = robotoc.JointTorquesLowerLimit(robot)
joint_torques_upper = robotoc.JointTorquesUpperLimit(robot)
constraints.add("joint_torques_lower", joint_torques_lower)
constraints.add("joint_torques_upper", joint_torques_upper)

# joint_velocity_lower = robotoc.JointVelocityLowerLimit(robot)
# joint_velocity_upper = robotoc.JointVelocityUpperLimit(robot)
# constraints.add("joint_velocity_lower", joint_velocity_lower)
# constraints.add("joint_velocity_upper", joint_velocity_upper)

# Create the OCP solver for unconstrained rigid-body systems.
T = 3
N = 100
ocp = robotoc.OCP(robot=robot, cost=cost, constraints=constraints, T=T, N=N)
solver_options = robotoc.SolverOptions()
solver_options.enable_benchmark = True
solver_options.nthreads = 4
solver_options.max_iter = 1000
ocp_solver = robotoc.UnconstrOCPSolver(ocp=ocp, solver_options=solver_options)

# Initial time and intial state
# t = 0.0
# q = np.array([-np.pi, 0])
# v = np.zeros(robot.dimv())

print("----- Solves the OCP by Riccati recursion algorithm. -----")
ocp_cputime=[]
for i in range (0,1000):
    t = 0.0
    q = np.array([-np.pi, 0])
    v = np.zeros(robot.dimv())
    ocp_solver.discretize(t)
    ocp_solver.set_solution("q", q)
    ocp_solver.set_solution("v", v)
    ocp_solver.init_constraints()
    ocp_solver.solve(t, q, v, init_solver=True)
    ocp_cputime.append(ocp_solver.get_solver_statistics().cpu_time)

ocpcputime_mean=np.mean(ocp_cputime)
ocpcputime_std =np.std(ocp_cputime)

print("KKT error after convergence: ", ocp_solver.KKT_error(t, q, v))
print(ocp_solver.get_solver_statistics())


# Solves the OCP by ParNMPC algorithm.
solver_options.nthreads = 8
solver_options.max_iter = 1000
parnmpc_solver = robotoc.UnconstrParNMPCSolver(ocp=ocp, solver_options=solver_options)

print("\n----- Solves the OCP by ParNMPC algorithm. -----")
par_cputime=[]
for i in range (0,10):
    t = 0.0
    q = np.array([-np.pi, 0])
    v = np.zeros(robot.dimv())
    parnmpc_solver.discretize(t)
    parnmpc_solver.set_solution("q", q)
    parnmpc_solver.set_solution("v", v)
    parnmpc_solver.init_constraints()
    parnmpc_solver.init_backward_correction()
    parnmpc_solver.solve(t, q, v, init_solver=True)
    par_cputime.append(parnmpc_solver.get_solver_statistics().cpu_time)

parcputime_mean=np.mean(par_cputime)
parcputime_std =np.std(par_cputime)
print("KKT error after convergence: ", parnmpc_solver.KKT_error(t, q, v))
print(parnmpc_solver.get_solver_statistics())

print(ocpcputime_mean,ocpcputime_std)
print(parcputime_mean,parcputime_std)



# visualization
rrq=ocp_solver.get_solution('q')
parq=parnmpc_solver.get_solution('q')
rrv=ocp_solver.get_solution('v')
parv=parnmpc_solver.get_solution('v')
rra=ocp_solver.get_solution('a')
para=parnmpc_solver.get_solution('a')
rru=ocp_solver.get_solution('u')
paru=parnmpc_solver.get_solution('u')
q_s=[[],[],[],[]]  
v_s=[[],[],[],[]]    
u_s=[[],[],[],[]]   
a_s=[[],[],[],[]]                    
for i in range(0,N):
    a=rrq[i]
    b=parq[i]
    q_s[0].append(a[0])
    q_s[1].append(a[1])
    q_s[2].append(b[0])
    q_s[3].append(b[1])
    c=rrv[i]
    d=parv[i]
    v_s[0].append(c[0])
    v_s[1].append(c[1])
    v_s[2].append(d[0])
    v_s[3].append(d[1])
    e=rru[i]
    f=paru[i]
    u_s[0].append(e[0])
    u_s[1].append(e[1])
    u_s[2].append(f[0])
    u_s[3].append(f[1])
    g=rra[i]
    h=para[i]
    a_s[0].append(g[0])
    a_s[1].append(g[1])
    a_s[2].append(h[0])
    a_s[3].append(h[1])
    #print(c)
plt.figure()
plt.plot(q_s[0],color='r',label='q0 RR')
plt.plot(q_s[1],color='r',label='q1 RR',     linestyle='--')
plt.plot(q_s[2],color='b',label='q0 PARNMPC')
plt.plot(q_s[3],color='b',label='q1 PARNMPC',linestyle='--')
plt.legend()
plt.title("trajectory of q\nT=1.0    N=1000")
plt.savefig("q.png")

plt.figure()
plt.plot(v_s[0],color='r',label='v0 RR')
plt.plot(v_s[1],color='r',label='v1 RR',     linestyle='--')
plt.plot(v_s[2],color='b',label='v0 PARNMPC')
plt.plot(v_s[3],color='b',label='v1 PARNMPC',linestyle='--')
plt.legend()
plt.title("trajectory of v\nT=1.0    N=1000")
plt.savefig("v.png")

plt.figure()
plt.plot(u_s[0],color='r',label='u0 RR')
plt.plot(u_s[1],color='r',label='u1 RR',     linestyle='--')
plt.plot(u_s[2],color='b',label='u0 PARNMPC')
plt.plot(u_s[3],color='b',label='u1 PARNMPC',linestyle='--')
plt.legend()
plt.title("trajectory of u\nT=1.0    N=1000")
plt.savefig("u.png")

plt.figure()
plt.plot(a_s[0],color='r',label='a0 RR')
plt.plot(a_s[1],color='r',label='a1 RR',     linestyle='--')
plt.plot(a_s[2],color='b',label='a0 PARNMPC')
plt.plot(a_s[3],color='b',label='a1 PARNMPC',linestyle='--')
plt.legend()
plt.title("trajectory of a\nT=1.0    N=1000")
plt.savefig("a.png")