"""
    Contains all code for running the CIDGIKc algorithm (i.e. constraints, SDP solvers, etc.).

    Usage: 
    Any call to our CIDGIKc.

    Author: <Hanna Zhang, hannajiamei.zhang@mail.utoronto.ca>
    Affl.: <University of Toronto, Department of Computer Science, Continuum Robotics Laboratory>
"""

from cidgikc.utils import rad2deg, angle_between
import numpy as np
import cvxpy as cp
import time
import mosek

class SdpSolverParams:
    """
    Parameters for cvxpy's SDP solvers (MOSEK, CVXOPT, and SCS).
    """
    def __init__(self, solver=cp.MOSEK, abstol=1e-1, reltol=1e-1, feastol=1e-1, max_iters=50, refinement_steps=10,
              kkt_solver='chol', alpha=10e-100000000000, scale=10e10000, normalize=True, use_indirect=True, qcp=False,
              mosek_params=None, feasibility=False, cost_function=None, verbose=True):
        self.solver = solver 
        self.abstol = abstol
        self.reltol = reltol
        self.feastol = feastol
        self.max_iters = max_iters
        self.qcp = qcp
        self.feasibility = feasibility
        self.verbose = verbose
        self.refinement_steps = refinement_steps
        self.kkt_solver = kkt_solver 
        self.alpha = alpha  
        self.scale = scale
        self.normalize = normalize 
        self.use_indirect = use_indirect
        self.cost_function = cost_function
        if mosek_params is None:
            self.mosek_params = {'MSK_IPAR_INTPNT_MAX_ITERATIONS': max_iters,
                                 'MSK_DPAR_INTPNT_TOL_PFEAS': abstol,
                                 'MSK_DPAR_INTPNT_TOL_DFEAS': abstol,
                                 'MSK_DPAR_INTPNT_TOL_REL_GAP': reltol,
                                 'MSK_DPAR_INTPNT_TOL_INFEAS': feastol,
                                 'MSK_IPAR_INFEAS_REPORT_AUTO': True,
                                 'MSK_IPAR_INFEAS_REPORT_LEVEL': 10,
                                 mosek.iparam.intpnt_scaling: mosek.scalingtype.free,
                                 mosek.iparam.intpnt_solve_form: mosek.solveform.primal,
                                 mosek.iparam.ana_sol_print_violated: mosek.onoffkey.on,
                                 mosek.dparam.intpnt_co_tol_near_rel: 1e10 
                                 }
        else:
            self.mosek_params = mosek_params
            
def ee_error(d, goal, soln):
    """ Returns the end-effector error.  """
    # INPUT:
    # d: problem dimension
    # goal: pose query/goal pose
    # soln: output pose
    # OUTPUT:
    # unit vector 
    position_err = np.linalg.norm(goal[0:d, d]-soln[0:d, d])
    rotation_err_1 = rad2deg(angle_between(goal[0:d, d-1], soln[0:d, d-1]))
    rotation_err_2 = rad2deg(angle_between(goal[0:d, 0], soln[0:d, 0]))
    return (position_err, rotation_err_1, rotation_err_2)
    
def CIDGIKc(Z_guess, base, baseprime, w_goal, wprime_goal, wpprime_goal, n, d, m, kmax, ee_spec, ee_scalefac, eigen_tol, L_range, obstacles=None):
    """ Runs the CIDGIKc algorithm.  """
    # INPUT:
    # Z_guess: lifted matrix variable represeting the straight mid-extension initialization
    # base, baseprime: base specification points
    # w_goal, wprime_goal, wpprime_goal: end-effector specification points
    # n: number of segments
    # d: dimension of problem
    # m: dimension of the lifted matrix variable
    # kmax: maximum allowable number of iterations
    # ee_spec: end-effector specification type
    # eigen_tol: eigenvalue limit to be considered converged
    # L_range: min and max extension of each segment
    # obstacles: tuple of obstacle information
    # OUTPUT:
    # Z.value: lifted matrix variable representing the inverse kinematics solution
    # eig_vals: eigenvalues throughout the solve
    # k: number of iterations to problem convergences
    # times: saved times taken for each iteration of CIDGIKc

    k = 0
    eig_vals = np.zeros((m,kmax))
    Z = cp.Variable((m,m), PSD=True)

    constraints = []
    if obstacles != None:
        constraints += get_obstacle_constraints(Z, n, d, obstacles)
    constraints += get_constraints(Z, n, d, m, ee_spec, base, baseprime, w_goal, wprime_goal, wpprime_goal, ee_scalefac, L_range)

    # C is a mxm matrix
    C = np.eye(m) # intialize the cost as identity
    C_param = cp.Parameter(shape=(m,m))
    objective = cp.Minimize(cp.trace(C_param@Z))
    prob = cp.Problem(objective, constraints)
    assert prob.is_dcp(dpp=True)
    solver_params = SdpSolverParams()
    
    times = []    
    Z.value = Z_guess
    
    while k<kmax:
        prob2solve = prob

        start = time.time()        
        C_param.value = C        
        if k==0:
            # give it an initial configuration and get the cost first
            C = conv_it_P5(Z.value, d)
            C_param.value = C
            prob2solve.solve(verbose=False, solver="MOSEK", mosek_params=solver_params.mosek_params, warm_start=True)
        else:
            prob2solve.solve(verbose=False, solver="MOSEK", mosek_params=solver_params.mosek_params, warm_start=True)

        if (prob.status == "optimal"):
            C = conv_it_P5(Z.value, d)

        end = time.time()
        times.append(end - start)

        # Save the eigenvalues 
        eig_vals[:,k] = sorted(np.linalg.eigvals(Z.value)[0:m])
        print("Iteration: ", k, "     EIGENVALUE: ", eig_vals[-(d+1),k])
        
        # We have a rank-d solution, eigenvalue is sufficently small, exit loop
        if eig_vals[-(d+1),k] < eigen_tol:
            k=k+1
            break
            
        k=k+1

    return (Z.value, eig_vals, k, times)

def get_obstacle_constraints(Z, n, d, obstacles):
    """ Returns quadratic contraints obstacle avoidance with segment endpoints.  """
    # INPUT:
    # Z: lifted matrix variable for IK solution
    # n: number of segments
    # d: dimension of problem
    # obstacles: tuple of obstacle information
    # OUTPUT:
    # obstacle_constraints: obstacle contraints to be used in solver

    obstacle_constraints = []
    all_radius = obstacles[0]

    if d==2:
        all_loc = np.delete(obstacles[1], 1, 1)
    else:
        all_loc = obstacles[1]

    obstacle_constraints = []
    
    # obstacle avoidance only with segment endpoints 
    for t in range(0,n):
        for o in range(0, all_radius.shape[0]):
            # prevent collisions with segment endpoints
            if t != n-1:
                obstacle_constraints += [ Z[2*t+1, 2*t+1] -2*Z[2*t+1, -d:]@all_loc[o] + all_loc[o].T@all_loc[o] >= all_radius[o]**2]
                    
                # prevent collisions with virtual joints
                # obstacle_constraints += [ Z[2*t, 2*t] -2*Z[2*t, -d:]@all_loc[o] + all_loc[o].T@all_loc[o] >= all_radius[o]**2]

    return obstacle_constraints
                
def get_constraints(Z, n, d, m, ee_spec, base, baseprime, w_goal, wprime_goal, wpprime_goal, ee_scalefac, L_range):
    """ Returns quadratic contraints defining segment triangles (i.e. our continuum robot).  """
    # INPUT:
    # Z: lifted matrix variable for IK solution
    # n: number of segments
    # d: dimension of problem
    # m: dimension of the lifted matrix variable
    # base, baseprime: base specification variables
    # w_goal, wprime_goal, wpprime_goal: end-effector specification variables
    # ee_scalefac: scaling of ee specification points
    # L_range: min and max extension of each segment
    # OUTPUT:
    # constraints: contraints to be used in solver

    # bottom right corner identity matrix
    identity_constraints = [Z[-d:, -d:] == np.eye(d)]
    
    scalar_constraints = [] # setup scalar values for segment continuity, i.e. omega_t values
    if ee_spec == 0:
        n_plus = n
    if ee_spec == 1 or ee_spec == 2:
        n_plus = n+1
        
    for t in range(0, n_plus): # make the diagonals of the identity matrices equal for the auxillary variable        
        ind = 2*(n-1)+1 + t*d # number of points
        scalar_constraints += [ Z[ind, -d] >= 0 ] # constrain the auxillary variable to be positive

        if d==3:
            # make the non-diagonal entries = 0 (along the right edge)
            scalar_constraints += [ Z[ind, -d+1] == 0, Z[ind, -d+2] == 0, Z[ind+1, -d+2] == 0 ]
            scalar_constraints += [ Z[ind+1, -d] == 0, Z[ind+2, -d] == 0, Z[ind+2, -d+1] == 0 ]
        if d==2:
            scalar_constraints += [ Z[ind, -d+1] == 0, Z[ind+1, -d] == 0 ]

        # make top right match all the other diagonal terms    
        for k in range(1,d):
            scalar_constraints += [ Z[ind, -d] == Z[ind+k, -d+k] ]
        
        scalar_constraints += [ Z[ind, ind] >= 0 ] # constrain the auxillary variable ^2 to be positive
        
        # make the non-diagonal entries = 0 (along the diagonal)
        if d==3:
            scalar_constraints += [ Z[ind, ind+1] == 0, Z[ind, ind+2] == 0, Z[ind+1, ind+2] == 0 ]
            scalar_constraints += [ Z[ind+1, ind] == 0, Z[ind+2, ind] == 0, Z[ind+2, ind+1] == 0 ]
        if d==2:
            scalar_constraints += [ Z[ind, ind+1] == 0, Z[ind+1, ind] == 0 ]

        # make the center diagonal squared terms match all the other diagonal terms 
        for k in range(1,d):
            scalar_constraints += [ Z[ind, ind] == Z[ind+k, ind+k] ]
            
    colinear_constraints = []
    
    # first segment is specified with base prime (point in the desired base orientation direction)
    scalar_constraints += [ Z[2*(n-1)+1, -d] >= (L_range[0,0]/2)/ee_scalefac]  # constrain the auxillary variable to be greater than the min
    colinear_constraints += [ Z[0, -d:] == base + ( Z[2*(n-1)+1, -d]*base - Z[2*(n-1)+1, -d]*baseprime ) ] # q1 = base + omega1(baseprime-base)
    
    if ee_spec==1: # final segment is specified with goal prime (5 dof pose)        
        # q_n = p_n + omega(n+1)(p_n - wprime_goal)        
        colinear_constraints += [ Z[2*(n-1),-d:] == w_goal + (Z[2*(n-1)+1 + n*d, -d]*w_goal - Z[2*(n-1)+1 + n*d, -d]*wprime_goal) ] # orientation
        scalar_constraints += [ Z[2*(n-1)+1 + n*d, -d] >= (L_range[0,n-1]/2)/ee_scalefac] # constrain the auxillary variable to be greater than the min

    elif ee_spec==2:  # final segment is specified with goal prime (6 dof pose) 
        # q_n = p_n + omega(n+1)(p_n - wprime_goal)        
        colinear_constraints += [ Z[2*(n-1),-d:] == w_goal + (Z[2*(n-1)+1 + n*d, -d]*w_goal - Z[2*(n-1)+1 + n*d, -d]*wprime_goal) ] # orientation
        scalar_constraints += [ Z[2*(n-1)+1 + n*d, -d] >= (L_range[0,n-1]/2)/ee_scalefac] # (L[n-1]/2)/ee_scalefac ] # constrain the auxillary variable to be greater than the min

        # Use Pythagoras
        # (wpprime_goal-q_n)^2 = (w_goal-q_n)^2 + (w_goal-wpprime_goal)^2 --> w_goal^2 - w_goal^t*q_n - w_goal^T*wpprime_goal + wpprime_goal^T*q_n = 0
        colinear_constraints += [ w_goal.T@w_goal - w_goal.T@Z[-d:, 2*(n-1)] - w_goal.T@wpprime_goal + wpprime_goal.T@Z[-d:, 2*(n-1)] == 0 ]
        
        # (wpprime_goal-p_n)^2 = (w_goal-p_n)^2 + (w_goal-wpprime_goal)^2 --> w_goal^2 - w_goal^t*p_n - w_goal^T*wpprime_goal + wpprime_goal^T*p_n = 0
        colinear_constraints += [ w_goal.T@w_goal - w_goal.T@Z[-d:, 2*(n-1)-1] - w_goal.T@wpprime_goal + wpprime_goal.T@Z[-d:, 2*(n-1)-1] == 0 ]
    
    # Collinear constraints between segments
    for t in range(1,n):
        ind = 2*(n-1)+1 + t*d
        # q_(t+1) = p_(t) + omega_t*(p_t-q_t)
        colinear_constraints += [ Z[2*t, -d:] == Z[2*(t-1)+1, -d:] + (Z[2*(t-1)+1, ind:ind+d] - Z[2*(t-1), ind:ind+d]) ]

    symmetry_constraints = []

    for t in range(0,n):
        # (p_(t-1)-q_t)^2 = (p_t-q_t)^2 --> (p_(t-1))^2 - 2*p_(t-1)^T*q_t = (p_t)^2 - 2*p_t^T*q_t        
        if t==0 and n==1: # single segment
            symmetry_constraints += [ base.T@base - 2*base.T@Z[-d:, 0] == w_goal.T@w_goal - 2*w_goal.T@Z[-d:, 0] ]
        elif t==0: # first segment
            symmetry_constraints += [ base.T@base - 2*base.T@Z[-d:, 0] == Z[1,1] - 2*Z[1, 0] ]
        elif t==n-1: # last segment 
            symmetry_constraints += [ Z[2*(t-1)+1, 2*(t-1)+1] - 2*Z[2*(t-1)+1, 2*t] == w_goal.T@w_goal - 2*w_goal.T@Z[-d:, 2*t] ]
        else: # middle segments
            symmetry_constraints += [ Z[2*(t-1)+1, 2*(t-1)+1] - 2*Z[2*(t-1)+1, 2*t] == Z[2*t+1, 2*t+1] - 2*Z[2*t+1, 2*t] ]

    length_constraints = []

    if not(n==1):
        for t in range(0,n): # fixed length segments
            # (2*Lt/pi)^2 <= (p_(t-1) - p_t)^2 <= (L_t)^2 --> (2*Lt/pi)^2 <= p_(t-1)^2 - 2*p_(t-1).T*p_t + p_t^2 <= (L_t)^2 
            if t==0: # first segment
                if ee_spec == 0:
                    length_constraints += [ base.T@base - 2*base.T@Z[-d:, 1] + Z[1, 1] <= (2*(L_range[1,t])/np.pi)**2 ]     
                length_constraints += [ base.T@base - 2*base.T@Z[-d:, 1] + Z[1, 1] >=  L_range[0,t]**2]         
            elif t==n-1: # last segment 
                if ee_spec == 0:
                    length_constraints += [ Z[2*t-1, 2*t-1] - 2*Z[2*t-1, -d:]@w_goal + w_goal.T@w_goal <= (2*( L_range[1,t])/np.pi)**2 ]    
                length_constraints += [ Z[2*t-1, 2*t-1] - 2*Z[2*t-1, -d:]@w_goal + w_goal.T@w_goal >=   L_range[0,t]**2]
            else: # middle segments
                length_constraints += [ Z[2*t-1, 2*t-1] - 2*Z[2*t-1, 2*t+1] + Z[2*t+1, 2*t+1] >=  L_range[0,t]**2]

    constraints = identity_constraints + scalar_constraints + colinear_constraints + symmetry_constraints + length_constraints 
    
    return constraints

def conv_it_P5(Z, d):
    """ Solves Problem 5.  """
    # INPUT:
    # Z: lifted matrix variable for IK solution
    # d: dimension of problem
    # OUTPUT:
    # C_star: updated cost

    _, Q = np.linalg.eigh(Z)
    Q = np.flip(Q, 1)
    U = Q[:, d:]
    C_star = U@U.transpose()
    
    return C_star