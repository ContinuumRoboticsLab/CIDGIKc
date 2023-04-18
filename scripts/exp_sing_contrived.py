"""
    Runs the CIDGIKc algorithm to solve IK for a single IK query obtained from a real robot configuration you specify the forward kinematics arc-parameters for.

    Usage: 
    python exp_sing_contrived.py 

    Author: <Hanna Zhang, hannajiamei.zhang@mail.utoronto.ca>
    Affl.: <University of Toronto, Department of Computer Science, Continuum Robotics Laboratory>
"""

from cidgikc.utils import forward_kinematics, CC2Triangle, Triangle2CC
from cidgikc.convex_iteration import CIDGIKc, ee_error

import numpy as np
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
np.random.seed(1331)

##########################################################################################################################
# PROBLEM PARAMETER SPECIFICATION

solve_with_obstacles = True

# Robot setup
n = 3 # total segments
d = 3 # dimension = 2, i.e. planar
# Solver Specifications
ee_spec = 1 # 0:position, 1: 5-DoF pose, 2: 6-DoF pose (only for d =3)
eigen_tol = 1e-7 # eigenvalue limit to be considered converged
kmax = 100 # maximum number of iterations

# Segment Length Specifications (identical segments)
seg_length_mid = 0.35 # mid-extension
seg_range = 0.2 # extension/compression range with respect to mid-extension
min_L = (seg_length_mid - seg_range)*np.ones(n)
max_L = (seg_length_mid + seg_range)*np.ones(n)
L_range = np.array((min_L, max_L))

# forward kinematics parameters used to obtain IK query
theta = np.array([np.pi/3, 0, np.pi/6])
if d==2:    
    delta = np.zeros(n) # 0 for PLANAR
else:
    delta = np.array([np.pi, 3/4*np.pi, 0]) # 0 for PLANAR
L = np.array([0.3, 0.35, 0.3])

# base definition (fixed don't change)
ee_scalefac = 1 # scaling of ee specification points
base_T = np.eye(d+1) # base as a transformation matrix
base =  base_T[0:d, d]
baseprime = -ee_scalefac * base_T[0:d, d-1]

# Populate with obstacles
if solve_with_obstacles:
    num_obs = 2 # ensure the number of obstacles here is consistent with "all_radius", "all_obs_color", "all_loc"

    all_radius = np.array([0.3, 0.3])
    all_obs_color = np.array(['red', 'red'])
    all_loc  = np.array( [[0, 0.0, 0.5], [0.6, 0.0, 0.8]] ) # always give x,y,z, for d=2 the code will delete the y entry

    all_ind = [i for i in range(0, num_obs)]
    obstacles = (all_radius, all_loc, all_obs_color, all_ind)
else:
    obstacles = None

##########################################################################################################################
# PROBLEM SETUP

# Run Forwards Kinematics to obtain IK query
w_goal, wprime_goal, wpprime_goal, ee_goal, all_CC_arcs = forward_kinematics(theta, delta, L, n, d, ee_scalefac)

# Setup a straight mid-extension initialization
L_guess = seg_length_mid*np.ones(n)
theta_guess = np.zeros(n)
delta_guess = np.zeros(n)
w_goal_guess, wprime_goal_guess, wpprime_goal_guess, T_c_tip_guess, all_CC_arcs_guess = forward_kinematics(theta_guess, delta_guess, L_guess, n, d, ee_scalefac)
(T, Z_guess) = CC2Triangle(L_guess, theta_guess, delta_guess, n, d, base_T, baseprime, w_goal_guess, wprime_goal_guess, wpprime_goal_guess, ee_spec)       
##########################################################################################################################
# SOLVING IK WITH CONVEX ITERATION

if ee_spec==0:
	m = 2*(n-1)+1 + n*d + d
elif ee_spec==1 or ee_spec==2:
	m = 2*(n-1)+1 + (n+1)*d + d
    
if solve_with_obstacles:
    (Z_soln, eig_vals, k, times) = CIDGIKc(Z_guess, base, baseprime, w_goal, wprime_goal, wpprime_goal, n, d, m, kmax, ee_spec, ee_scalefac, eigen_tol, L_range, obstacles=obstacles)
else:
    (Z_soln, eig_vals, k, times) = CIDGIKc(Z_guess, base, baseprime, w_goal, wprime_goal, wpprime_goal, n, d, m, kmax, ee_spec, ee_scalefac, eigen_tol, L_range)

print("Solved in ", np.sum(times), " seconds.")
print("Completed in ", k, "iterations.")

##########################################################################################################################
# SOLUTION RECOVERY

theta_soln, delta_soln, L_soln  = Triangle2CC(Z_soln, n, d, base, baseprime, w_goal, wprime_goal, wpprime_goal)
soln_params = (theta_soln, delta_soln, L_soln)

print("theta_soln [rad] : ", theta_soln)
print("delta_soln [rad] : ", delta_soln)
print("L_soln [m]       : ", L_soln)

w_goal_recov, wprime_goal_recov, wpprime_goal_recov, T_c_tip, all_CC_arcs = forward_kinematics(theta_soln, delta_soln, L_soln, n, d, ee_scalefac)

ee_res = np.reshape(T_c_tip, (d+1, d+1), order='F')
ee_goal = np.reshape(ee_goal, (d+1, d+1), order='F')

(position_err, rotation_err_1, rotation_err_2) = ee_error(d, ee_goal, ee_res)

if ee_spec == 0:
    print("position err [m]: ", position_err)            
elif ee_spec == 1:
    print("position err [m]   : ", position_err)             
    print("rotation z err[deg]: ", rotation_err_1)            
elif ee_spec == 2:
    print("position err [m]    : ", position_err)             
    print("rotation z err [deg]: ", rotation_err_1)      
    print("rotation y err [deg]: ", rotation_err_2)     
