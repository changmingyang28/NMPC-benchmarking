import robotoc
import numpy as np
import math

import pinocchio as pin
from pinocchio.visualize import GepettoVisualizer


model_info = robotoc.RobotModelInfo()
model_info.urdf_path = "robotoc/iiwa_description/urdf/iiwa14.urdf"
robot = robotoc.Robot(model_info)

mesh_dir = "robotoc/iiwa_description/meshes/iiwa14"
# load robot model and visuals

model, collision_model, visual_model = pin.buildModelsFromUrdf(
                model_info.urdf_path,
                mesh_dir
            )
viz_manipulator =  GepettoVisualizer(model, collision_model, visual_model)
viz_manipulator.initViewer()
viz_manipulator.loadViewerModel("robot")
viz_manipulator.display(np.array([0,0,0,0,0,0,0]))
input()
# Change the limits from the default parameters.
robot.set_joint_effort_limit(np.full(robot.dimu(), 50))
robot.set_joint_velocity_limit(np.full(robot.dimv(), 0.5 * math.pi))
           
# Create a cost function.
cost = robotoc.CostFunction()
config_cost = robotoc.ConfigurationSpaceCost(robot)
q_ref = np.array([0, 0.5 * math.pi, 0, 0.5 * math.pi, 0, 0.5 * math.pi, 0])
config_cost.set_q_ref(q_ref)
config_cost.set_q_weight(np.full(robot.dimv(), 10))
config_cost.set_q_weight_terminal(np.full(robot.dimv(), 10))
config_cost.set_v_weight(np.full(robot.dimv(), 0.01))
config_cost.set_v_weight_terminal(np.full(robot.dimv(), 0.01))
config_cost.set_a_weight(np.full(robot.dimv(), 0.01))
cost.add("config_cost", config_cost)

# Create joint constraints.
constraints = robotoc.Constraints(
    barrier_param=1.0e-03, fraction_to_boundary_rule=0.995
)
joint_position_lower = robotoc.JointPositionLowerLimit(robot)
joint_position_upper = robotoc.JointPositionUpperLimit(robot)
joint_velocity_lower = robotoc.JointVelocityLowerLimit(robot)
joint_velocity_upper = robotoc.JointVelocityUpperLimit(robot)
joint_torques_lower = robotoc.JointTorquesLowerLimit(robot)
joint_torques_upper = robotoc.JointTorquesUpperLimit(robot)
constraints.add("joint_position_lower", joint_position_lower)
constraints.add("joint_position_upper", joint_position_upper)
constraints.add("joint_velocity_lower", joint_velocity_lower)
constraints.add("joint_velocity_upper", joint_velocity_upper)
constraints.add("joint_torques_lower", joint_torques_lower)
constraints.add("joint_torques_upper", joint_torques_upper)

# Create the OCP solver for unconstrained rigid-body systems.
T = 3.0
N = 60
ocp = robotoc.OCP(robot=robot, cost=cost, constraints=constraints, T=T, N=N)
solver_options = robotoc.SolverOptions()
solver_options.nthreads = 4
ocp_solver = robotoc.UnconstrOCPSolver(ocp=ocp, solver_options=solver_options)

# Initial time and intial state
t = 0.0
q = np.array([0.5 * math.pi, 0, 0.5 * math.pi, 0, 0.5 * math.pi, 0, 0.5 * math.pi])
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

