"""
    Contains helper functions to support the CIDGIKc framework.

    Author: <Hanna Zhang, hannajiamei.zhang@mail.utoronto.ca>
    Affl.: <University of Toronto, Department of Computer Science, Continuum Robotics Laboratory>
"""

import numpy as np
import math

def rad2deg(rad):
    ''' Convert radians to degrees. '''
    # INPUT: 
    # rad: radians
    # OUTPUT: 
    # deg: degrees
    
    # 1Rad × 180/π = 57.296Deg
    return rad * 180 / np.pi
    
def forward_kinematics(theta, delta, L, n, d, ee_scalefac):
    ''' Run the forward constant curvature kinematics. '''
    # INPUT: 
    # theta, delta, L: contant curvature arc parameters
    # n: number of segments
    # d: problem dimension
    # ee_scalefac: scaling factor to place end-effector specification points
    # OUTPUT: 
    # w_goal, wprime_goal, wpprime_goal: end-effector specification points
    # T_c_tip: end-effector pose
    # all_CC_arcs: robot body information (points along the curve)

    all_CC_arcs = np.zeros((500, (d+1)**2, n)) # discretization of a CC arc

    for i in range(0, n):
        if i == 0:  # section 1
            T, T_tip = trans_mat_cc(theta[i] / L[i], delta[i], L[i], d)
            all_CC_arcs[:,:,i] = T
            T_c_tip = T_tip
        else:
            T_next, T_tip_next = trans_mat_cc(theta[i] / L[i], delta[i], L[i], d)
            T_c, T_c_tip = couple_transforms(T_next, T_tip, d)
            all_CC_arcs[:,:,i] = T_c
            T_tip = T_c_tip
            
    T_goal = np.reshape(T_c_tip, (d+1,d+1), order='F')
    # target position & orientation of EE
    p_star = T_goal[0:d, d].transpose()  
    # current pose of EE
    zhat_star = T_goal[0:d, d-1].transpose()  # desired orientation SO(3)
    xhat_star = T_goal[0:d, d-2].transpose()  # desired orientation SE(3)    
    
    w_goal = p_star
    wprime_goal = p_star + ee_scalefac*zhat_star
    wpprime_goal = p_star + ee_scalefac*xhat_star

    return w_goal, wprime_goal, wpprime_goal, T_c_tip, all_CC_arcs

def CC2Triangle(L, theta, delta, n, d, base_T, baseprime, w_goal, wprime_goal, wpprime_goal, ee_spec):
    """ Converts constant curvature representation of a continuum robot to graph representation."""
    # INPUT:
    # L, theta, delta: constnat curvature representation arc parameters 
    # n: number of segments
    # d: problem dimension
    # base_T: base pose
    # base, baseprime: base specification points
    # w_goal, wprime_goal, wpprime_goal: goal specification points
    # ee_spec: end-effector specification
    # OUTPUT:
    # T: end-effector pose
    # Z: lifted matrix variable containing graph information (i.e. segment triangle points)

    base = np.reshape(base_T[:d, d], (d), order='F')
    l = np.zeros((n, 1))  # virtual joint lengths
    q_j = np.zeros((n, d))  # positions of joints

    Rtot = np.eye(d);  # orientation at each segment end
    te = np.zeros((d, 1))  # positions of each segment end
    T = np.eye(d+1)

    X = np.zeros((2 * (n - 1) + 1, d))
    w = np.zeros((n, 1))

    for t in range(0, n):
        if theta[t] == 0:
            l[t] = L[t] / 2
        else:
            l[t] = L[t] / theta[t] * np.tan(theta[t] / 2)

        if d == 2:
            if t == 0:  # base segment
                q_j[t, :] = np.array([0, 0]) + l[t] * np.array([0, 1])
            else:
                q_j[t, :] = q_j[t - 1, :] + (Rtot @ ((l[t] + l[t - 1]) * np.array([0, 1]).T).T)
        if d == 3:
            if t == 0:  # base segment
                q_j[t, :] = np.array([0, 0, 0]) + l[t] * np.array([0, 0, 1])
            else:
                q_j[t, :] = q_j[t - 1, :] + (Rtot @ ((l[t] + l[t - 1]) * np.array([0, 0, 1]).T).T)

        # Couple transforms
        if t == 0:
            T, T_tip = trans_mat_cc(theta[t] / L[t], delta[t], L[t], d)
            T_c_tip = T_tip
        else:
            T_next, T_tip_next = trans_mat_cc(theta[t] / L[t], delta[t], L[t], d)
            T_c, T_c_tip = couple_transforms(T_next, T_tip, d)
            T_tip = T_c_tip
            
        T_c_tip_tmp = np.reshape(T_c_tip, (d+1,d+1), order='F')

        Rtot = T_c_tip_tmp[0:d, 0:d]
        te = T_c_tip_tmp[0:d, d]

        # ASSIGN ct, qt, pt for this segment
        if t == 0 and n == 1:  # only one segment
            X[t * 2, :] = q_j[t, :]  # pt-1
        elif t == 0:  # the first segment
            X[t * 2, :] = q_j[t, :]  # qt
            X[t * 2 + 1, :] = te  # pt
        elif t == n - 1:  # the last segment
            X[t * 2, :] = q_j[t, :]  # qt
        else:  # not the last segment
            X[t * 2, :] = q_j[t, :]  # qt
            X[t * 2 + 1, :] = te  # pt

        # SOLVE FOR W
        if t == 0 and n == 1:  # only one segment
            ptminus1 = base
            qt = X[t * 2, :]
            w_tmp = np.divide((qt - ptminus1), (base - baseprime))
            w_tmp = w_tmp[~np.isnan(w_tmp)]
            w_tmp = w_tmp[~(np.isinf(w_tmp))]
            w[t] = w_tmp @ (w_tmp != 0).T / np.sum((w_tmp != 0))
        elif t == 0:  # first segment
            ptminus1 = base
            qt = X[t * 2, :]
            w_tmp = np.divide((qt - base), (base - baseprime))
            w_tmp = w_tmp[~np.isnan(w_tmp)]
            w_tmp = w_tmp[~(np.isinf(w_tmp))]
            w[t] = w_tmp @ (w_tmp != 0).T / np.sum((w_tmp != 0))
        elif t == n - 1:  # last segment
            ptminus1 = X[(t - 1) * 2 + 1, :]
            qtminus1 = X[(t - 1) * 2, :]
            qt = X[t * 2, :]
            w_tmp = np.divide((qt - ptminus1), (ptminus1 - qtminus1))
            w_tmp = w_tmp[~np.isnan(w_tmp)]
            w_tmp = w_tmp[~(np.isinf(w_tmp))]
            w[t] = w_tmp @ (w_tmp != 0).T / np.sum((w_tmp != 0))
        else:  # internal segment
            ptminus1 = X[(t - 1) * 2 + 1, :]
            qtminus1 = X[(t - 1) * 2, :]
            qt = X[t * 2, :]
            w_tmp = np.divide((qt - ptminus1), (ptminus1 - qtminus1))
            w_tmp = w_tmp[~np.isnan(w_tmp)]
            w_tmp = w_tmp[~(np.isinf(w_tmp))]
            w[t] = w_tmp @ (w_tmp != 0).T / np.sum((w_tmp != 0))

    # define end effector pose
    T[0:d, 0:d] = Rtot
    T[0:d, d] = te

    # CONSTRUCT Z
    if ee_spec == 0:
        m = 2 * (n - 1) + 1 + n * d + d
    elif ee_spec == 1 or ee_spec == 2:
        m = 2 * (n - 1) + 1 + (n + 1) * d + d

    Z = np.zeros((m, m))
    eye = np.eye(d)
    X_bar = X.T

    for t in range(0, n):
        X_bar = np.hstack((X_bar, w[t] * eye))

    if ee_spec == 1 or ee_spec == 2:
        qn = X[2 * (n - 1), :]
        # q_n = p_n + omega(n+1)(p_n - wprime_goal)
        w_tmp = np.divide((w_goal - qn), (wprime_goal - w_goal))
        w_tmp = w_tmp[~(np.isnan(w_tmp))]
        w_tmp = w_tmp[~(np.isinf(w_tmp))]
        w_ee = w_tmp @ (w_tmp != 0).T / np.sum((w_tmp != 0))
        X_bar = np.hstack((X_bar, w_ee * eye))

    X_bar = np.hstack((X_bar, eye))
    Z = X_bar.T @ X_bar

    return (T, Z)

def Triangle2CC(Z, n, d, base, baseprime, w_goal, wprime_goal, wpprime_goal):
    """ Converts graph representation to constant curvature representation of a continuum robot. """
    # INPUT:
    # Z: lifted matrix variable containing graph information (i.e. segment triangle points)
    # n: number of segments
    # d: problem dimension
    # base, baseprime: base specification points
    # w_goal, wprime_goal, wpprime_goal: goal specification points
    # OUTPUT:
    # theta_all, delta_all, L_all: constant curvature arc representation variables
    
    theta_all = np.zeros(n)
    delta_all = np.zeros(n)
    L_all = np.zeros(n)
    normal = 0
    
    for t in range(0,n):         
        prev_CC_vars = (theta_all, delta_all, L_all)
        theta, delta, L, normal = Triangle2CC_singseg(Z, t+1, n, d, normal, prev_CC_vars, base, baseprime, w_goal, wprime_goal, wpprime_goal)
        
        theta_all[t] = theta
        delta_all[t] = delta
        L_all[t] = L
           
    return theta_all, delta_all, L_all


def Triangle2CC_singseg(Z, t, n, d, normal_prev, prev_CC_vars, base, baseprime, w_goal, wprime_goal, wpprime_goal):
    """ Helper function to "Triangle2CC". """
    # INPUT:
    # Z: lifted matrix variable containing graph information (i.e. segment triangle points)
    # t: segment index, starting at 1
    # n: number of segments
    # d: problem dimension
    # normal_prev: normal of the previous segment
    # base, baseprime: base specification points
    # w_goal, wprime_goal, wpprime_goal: goal specification points
    # OUTPUT:
    # theta_all, delta_all, L_all: constant curvature arc representation variables

    q_opt = Z[-d:, 2 * (t - 1)]
    
    # if this is the first segment 
    if t == 1:
        pminus1_opt = base
    else:
        pminus1_opt = Z[-d:, 2 * (t - 2) + 1]

    if t == n:
        p_opt = w_goal
    else:
        p_opt = Z[-d:, 2 * (t - 1) + 1]

    lt = (np.linalg.norm(q_opt - pminus1_opt) + np.linalg.norm(q_opt - p_opt))/2 # average the two joints to get the segment length
    v1 = q_opt - pminus1_opt
    v2 = p_opt - q_opt
    normal = np.cross(v1, v2)
    theta = angle_between(v1, v2)

    if theta==0:
        Lt = lt*2
    else:
        Lt = lt * theta / np.tan(theta / 2)
    
    if d==2: # for planar
        delta = 0
        theta_prev, delta_prev, L_prev = prev_CC_vars
        best_err = np.inf
        for case in range(0,2):
            L_prev[t-1] = Lt
            
            if case == 0:
                theta_prev[t-1] = theta
                delta_prev[t-1] = delta
            if case == 1:
                theta_prev[t-1] = -theta
                delta_prev[t-1] = delta
                
            ___, ___, ___, T_c_tip_curr, ___ = forward_kinematics(theta_prev, delta_prev, L_prev, t, d, 0.05)
            T_c_tip_curr = np.reshape(T_c_tip_curr,(d+1,d+1), order='F')
            
            if np.linalg.norm(T_c_tip_curr[0:d, d]-p_opt) < best_err:
                best_err = np.linalg.norm(T_c_tip_curr[0:d, d]-p_opt)
                theta_best = theta_prev[t-1]
                delta_best = delta_prev[t-1] 
        
        if best_err == np.inf:
            theta_best = theta_prev[t-1]
            delta_best = delta_prev[t-1] 
            
    elif d==3: # for spatial
        if theta==0:
            delta = 0
        elif t==1:
            delta = angle_between(np.array((0,1,0)), normal)
        else: 
            delta = angle_between(normal_prev, normal)
    
        theta_prev, delta_prev, L_prev = prev_CC_vars

        best_err = np.inf
        for case in range(0,20):
            L_prev[t-1] = Lt
            
            if case == 0:
                theta_prev[t-1] = theta
                delta_prev[t-1] = delta
            if case == 1:
                theta_prev[t-1] = -theta
                delta_prev[t-1] = delta
            if case == 2:
                theta_prev[t-1] = theta
                delta_prev[t-1] = -delta
            if case == 3:
                theta_prev[t-1] = -theta
                delta_prev[t-1] = -delta
            if case == 4:
                theta_prev[t-1] = theta
                delta_prev[t-1] = -delta + np.pi
            if case == 5:
                theta_prev[t-1] = theta
                delta_prev[t-1] = -delta + np.pi
            if case == 6:
                theta_prev[t-1] = theta
                delta_prev[t-1] = delta + np.pi
            if case == 7:
                theta_prev[t-1] = theta
                delta_prev[t-1] = delta - np.pi
            if case == 8:
                theta_prev[t-1] = theta
                delta_prev[t-1] = -delta - np.pi
            if case == 9:
                theta_prev[t-1] = theta
                delta_prev[t-1] = -delta - np.pi
            if case == 10:
                theta_prev[t-1] = theta
                delta_prev[t-1] = delta - np.pi
            if case == 11:
                theta_prev[t-1] = theta
                delta_prev[t-1] = delta - np.pi
            if case == 12:
                theta_prev[t-1] = -theta
                delta_prev[t-1] = -delta + np.pi
            if case == 13:
                theta_prev[t-1] = -theta
                delta_prev[t-1] = -delta + np.pi
            if case == 14:
                theta_prev[t-1] = -theta
                delta_prev[t-1] = delta + np.pi
            if case == 15:
                theta_prev[t-1] = -theta
                delta_prev[t-1] = delta - np.pi
            if case == 16:
                theta_prev[t-1] = -theta
                delta_prev[t-1] = -delta - np.pi
            if case == 17:
                theta_prev[t-1] = -theta
                delta_prev[t-1] = -delta - np.pi
            if case == 18:
                theta_prev[t-1] = -theta
                delta_prev[t-1] = delta - np.pi
            if case == 19:
                theta_prev[t-1] = -theta
                delta_prev[t-1] = delta - np.pi
                
            ___, ___, ___, T_c_tip_curr, ___ = forward_kinematics(theta_prev, delta_prev, L_prev, t, d, 0.05)
            T_c_tip_curr = np.reshape(T_c_tip_curr,(d+1,d+1), order='F')
            
            if np.linalg.norm(T_c_tip_curr[0:3, 3]-p_opt) < best_err:
                best_err = np.linalg.norm(T_c_tip_curr[0:3, 3]-p_opt)
                theta_best = theta_prev[t-1]
                delta_best = delta_prev[t-1] 
        
        if best_err == np.inf:
            theta_best = theta_prev[t-1]
            delta_best = delta_prev[t-1] 

    theta = theta_best 
    delta = delta_best

    return theta, delta, Lt, normal

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    # INPUT:
    # vector: any dimension vection
    # OUTPUT:
    # unit vector 
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'. """
    # INPUT:
    # v1, v2: two vectors
    # OUTPUT:
    # angle: the angle between the two vectors

    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    
    if (np.dot(v1_u, v2_u) >= 1):
        dot_prod = 1
    else:
        dot_prod = np.dot(v1_u, v2_u)

    angle = np.arccos(dot_prod)  
    
    if np.isnan(angle):
        angle = 0
    
    return angle

def trans_mat_cc(kappa, phi, l, d): 
    ''' Get transformation matrices along constant curvature segment. '''
    # INPUT:
    # kappa, phi, l: constant curvature arc parameters
    # d: dimension of problem
    # OUTPUT:
    # T: array of flattened transformation matrices
    # T_tip: segment end pose

    sect_points=500
    si = np.linspace(0,l,sect_points)
    T = np.zeros((len(si), 16))
    
    c_p = float(np.cos(-phi))
    s_p = float(np.sin(-phi))
    
    for i in range(0,len(si)):
        s = si[i]
        c_ks = float(np.cos(kappa*s))
        s_ks = float(np.sin(kappa*s))
        kappa = float(kappa)
        if kappa == 0:
            T[i,0:16] = np.array([c_p*c_ks, s_p*c_ks, -s_ks, 0.0, -s_p, c_p, 0.0, 0.0, c_p*s_ks, s_p*s_ks, c_ks, 0.0, 0.0, 0.0, s, 1.0])
        else:
            T[i,0:16] = np.array([c_p*c_ks, s_p*c_ks, -s_ks, 0.0, -s_p, c_p, 0.0, 0.0, c_p*s_ks, s_p*s_ks, c_ks, 0.0, (c_p*(1.0-c_ks))/kappa, (s_p*(1.0-c_ks))/kappa, s_ks/kappa, 1.0])
            
    if d==2:
        T = np.delete(T, np.s_[1::4], 1)
        T = np.delete(T, np.s_[3:6:], 1) 
    T_tip = T[-1, :]
    
    return (T, T_tip)

def couple_transforms(T, T_tip, d):
    ''' Find orientation and position of distal section, multiply T of current section with T at tip of previous section. '''
    # INPUT:
    # T: transformation matrices of current section
    # T_tip: transformation at tip of previous section
    # OUTPUT:
    # Tc: coupled transformation matrix
    # Tc_tip: last line of Tc (section end)
    
    T_c = np.zeros((len(T[:,1]),len(T[1,:])))
    T_tip = np.reshape(T_tip,(d+1,d+1), order='F')
    
    for k in range(0,len(T[:,1])):
        T_c[k,:] = np.reshape(T_tip@np.reshape(T[k,:], (d+1, d+1), order='F'), ((d+1)**2), order='F')
    T_c_tip = T_c[-1,:]
    
    return (T_c, T_c_tip)
