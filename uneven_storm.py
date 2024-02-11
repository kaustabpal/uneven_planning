import torch
import torch.nn.functional as F
import ghalton
import numpy as np
# np.random.seed(42)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import copy
import time
from scipy.interpolate import BSpline
from scipy.optimize import fsolve, root
import scipy.interpolate as si
import theseus as th
import os
np.set_printoptions(suppress=True)

class Goal_Sampler:
    def __init__(self, c_state, g_state, vl, wl, obstacles):
        '''
        c_state: current_state
        g_state: goal_state
        vl: last velocity
        wl: last angular velocity
        obstacles: list of obstacles
        '''
        # agent info
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.radius = 1.0
        self.c_state = c_state # start state
        self.g_state = g_state # goal state
        self.step_size_mean = 0.9
        self.step_size_cov = 0.7
        self.avoid_obs = False
        self.vl = vl
        self.wl = wl
        self.v_ub = 10
        self.v_lb = 1
        self.w_ub = 0.52 # 45degs/sec
        self.w_lb = -0.52 # -45degs/sec
        self.max_ctrl = torch.tensor([self.v_ub, self.w_ub])
        self.min_ctrl = torch.tensor([self.v_lb, self.w_lb])
        self.amin = -3.19
        self.amax = 3.19
        self.jmin = -0.1
        self.jmax = 0.1
        self.init_q = [self.vl, self.wl]

        # obstacle info
        self.obst_radius = 1.0
        self.obstacles = obstacles
        self.n_obst = 0
        
        # MPC params
        # self.N = 2 # Number of samples
        self.dt = 0.05
        self.horizon = 30 # Planning horizon
        self.d_action = 2 # dimension of action space
        # # For smoothening the trajectories
        self.knot_scale = 4
        self.n_knots = self.horizon//self.knot_scale
        self.ndims = self.n_knots*self.d_action
        self.bspline_degree = 2
        self.num_particles = 100
        self.top_K = int(0.1*self.num_particles) # Number of top samples
        self.null_act_frac = 0.01
        self.num_null_particles = round(int(self.null_act_frac * self.num_particles * 1.0))
        self.num_neg_particles = round(int(self.null_act_frac * self.num_particles)) -\
                                                            self.num_null_particles
        self.num_nonzero_particles = self.num_particles - self.num_null_particles -\
                                                            self.num_neg_particles
        self.sample_shape =  self.num_particles - 1

        if(self.num_null_particles > 0):
            self.null_act_seqs = torch.zeros(self.num_null_particles, self.horizon,\
                                                                    self.d_action)
        # self.initialize_mu()
        # self.initialize_sig()

        # Sampling params
        self.perms = ghalton.EA_PERMS[:self.ndims]
        self.sequencer = ghalton.GeneralizedHalton(self.perms)

        # init_q = torch.tensor(self.c_state)
        self.init_action = torch.zeros((self.horizon, self.d_action)) + torch.tensor(self.init_q)
        self.init_mean = self.init_action 
        self.mean_action = self.init_mean.clone()
        self.best_action = self.mean_action.clone()
        self.init_v_cov = 0.9
        self.init_w_cov = 0.9
        self.init_cov_action = torch.tensor([self.init_v_cov, self.init_w_cov])
        self.cov_action = self.init_cov_action
        self.scale_tril = torch.sqrt(self.cov_action)
        self.full_scale_tril = torch.diag(self.scale_tril)
        
        self.gamma = 0.99
        self.gamma_seq = torch.cumprod(torch.tensor([1]+[self.gamma]*(self.horizon-1)),dim=0).reshape(1,self.horizon)

        self.traj_N = torch.zeros((self.num_particles, self.horizon+1, 18))
        self.controls_N = torch.zeros((self.num_particles, self.horizon, 2))

        self.top_trajs = torch.zeros((self.top_K, self.horizon+1, 18))
        x_ten = torch.randn(self.num_particles*(self.horizon+1),1)
        y_ten = torch.randn(self.num_particles*(self.horizon+1),1)
        theta_ten = torch.randn(self.num_particles*(self.horizon+1),1)
        self.x = th.Variable(x_ten, name="x")
        self.y = th.Variable(y_ten, name="y")
        self.theta = th.Variable(theta_ten, name="theta")

        # opt_vars
        self.z = th.Vector(tensor=torch.rand(self.num_particles*(self.horizon+1), 1), name="z")
        self.beta = th.Vector(tensor=torch.rand(self.num_particles*(self.horizon+1), 1), name="beta")
        self.gamma = th.Vector(tensor=torch.rand(self.num_particles*(self.horizon+1), 1), name="gamma")

        self.x1 = th.Vector(tensor=torch.rand(self.num_particles*(self.horizon+1), 1), name="x1")
        self.y1 = th.Vector(tensor=torch.rand(self.num_particles*(self.horizon+1), 1), name="y1")
        self.z1 = th.Vector(tensor=torch.rand(self.num_particles*(self.horizon+1), 1), name="z1")

        self.x2 = th.Vector(tensor=torch.rand(self.num_particles*(self.horizon+1), 1), name="x2")
        self.y2 = th.Vector(tensor=torch.rand(self.num_particles*(self.horizon+1), 1), name="y2")
        self.z2 = th.Vector(tensor=torch.rand(self.num_particles*(self.horizon+1), 1), name="z2")

        self.x3 = th.Vector(tensor=torch.rand(self.num_particles*(self.horizon+1), 1), name="x3")
        self.y3 = th.Vector(tensor=torch.rand(self.num_particles*(self.horizon+1), 1), name="y3")
        self.z3 = th.Vector(tensor=torch.rand(self.num_particles*(self.horizon+1), 1), name="z3")

        self.x4 = th.Vector(tensor=torch.rand(self.num_particles*(self.horizon+1), 1), name="x4")
        self.y4 = th.Vector(tensor=torch.rand(self.num_particles*(self.horizon+1), 1), name="y4")
        self.z4 = th.Vector(tensor=torch.rand(self.num_particles*(self.horizon+1), 1), name="z4")

        self.optim_vars = self.z, self.beta, self.gamma, self.x1,self.y1,self.z1, self.x2,self.y2,self.z2, self.x3,self.y3,self.z3, self.x4,self.y4,self.z4
        self.aux_vars = self.x, self.y, self.theta
        self.cost_function = th.AutoDiffCostFunction(
            self.optim_vars, self.error_fn, 18, cost_weight=th.ScaleCostWeight(torch.tensor(1.0)), aux_vars=self.aux_vars, name="error")
        self.objective = th.Objective()
        self.objective.add(self.cost_function)
        self.optimizer = th.LevenbergMarquardt(
            self.objective,
            max_iterations=100,
            step_size=1,
        )
        self.theseus_optim = th.TheseusLayer(self.optimizer)
        self.theseus_optim.to(device=self.device)
        # self.top_traj = self.c_state.reshape(1,3)*torch.ones((self.horizon+1, 18))
        
        # self.device = "cpu" #torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # print(self.device)
        # self.curr_state_N = np.zeros((self.N,1,3))
        # self.V_N_T = np.zeros((self.N, self.horizon))
        # self.W_N_T = np.zeros((self.N, self.horizon))
 
    # def initialize_mu(self): # tensor contain initialized values'''
    #      self.MU = 0*torch.ones((2,self.horizon)) # 2 dim Mu for vel and Angular velocity
    
    # def initialize_sig(self):
    #     self.SIG = 0.7*torch.ones((2,self.horizon))
    def error_fn(self, optim_vars, aux_vars): # aux_vars=args
        #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # device = "cpu"
        k1 = (2.5-1)/np.abs(2.5-1) # for i=1
        k2 = (2.5-2)/np.abs(2.5-2) # for i=2
        k3 = (2.5-3)/np.abs(2.5-3) # for i=3
        k4 = (2.5-4)/np.abs(2.5-4) # for i=4
        h = 0.512/2 
        w = 0.5708/2 
        lamda = 2
        a = 0.05
        # l1,l2,l3,l4 = 0.003282,0.003282,0.003282,0.003282 
        l1,l2,l3,l4 = 0.03282,0.03282,0.03282,0.03282 
        # l1,l2,l3,l4 = 0.1651, 0.1651, 0.1651, 0.1651
        x, y, alpha = aux_vars #args[0], args[1], args[2]
        
        z = optim_vars[0]
        beta = optim_vars[1]
        gamma = optim_vars[2]
        
        x1 = optim_vars[3]
        y1 = optim_vars[4]
        z1 = optim_vars[5]
        
        x2 = optim_vars[6]
        y2 = optim_vars[7]
        z2 = optim_vars[8]
        
        x3 = optim_vars[9]
        y3 = optim_vars[10]
        z3 = optim_vars[11]
        
        x4 = optim_vars[12]
        y4 = optim_vars[13]
        z4 = optim_vars[14]
        
        eqn1 = (x.tensor + h*torch.cos(alpha.tensor)*torch.cos(beta.tensor) + k1*w*(torch.cos(alpha.tensor)*torch.sin(beta.tensor)*torch.sin(gamma.tensor) - torch.sin(alpha.tensor)*torch.cos(gamma.tensor)) - l1*(torch.cos(alpha.tensor)*torch.sin(beta.tensor)*torch.cos(gamma.tensor) - torch.sin(alpha.tensor)*torch.sin(gamma.tensor))-x1.tensor)
        eqn2 = (y.tensor + h*torch.sin(alpha.tensor)*torch.cos(beta.tensor) + k1*w*(torch.cos(alpha.tensor)*torch.sin(beta.tensor)*torch.sin(gamma.tensor) + torch.cos(alpha.tensor)*torch.cos(gamma.tensor)) - l1*(torch.sin(alpha.tensor)*torch.sin(beta.tensor)*torch.cos(gamma.tensor) - torch.cos(alpha.tensor)*torch.sin(gamma.tensor))-y1.tensor)
        eqn3 = (z.tensor - h*torch.sin(beta.tensor) + k1*w*torch.cos(beta.tensor)*torch.sin(gamma.tensor) - l1*torch.cos(beta.tensor)*torch.cos(gamma.tensor) - z1.tensor)
        eqn4 = (z1.tensor - lamda*torch.exp((-a*x1.tensor**2)+(-a*y1.tensor**2))) 
        # eqn4 = (z1.tensor - (0.5*torch.sin(x1.tensor) + 0.5*torch.cos(y1.tensor)))
        # eqn4 = z1.tensor - (x1.tensor+y1.tensor) 
        
        eqn5 = (x.tensor - h*torch.cos(alpha.tensor)*torch.cos(beta.tensor) + k2*w*(torch.cos(alpha.tensor)*torch.sin(beta.tensor)*torch.sin(gamma.tensor) - torch.sin(alpha.tensor)*torch.cos(gamma.tensor)) - l2*(torch.cos(alpha.tensor)*torch.sin(beta.tensor)*torch.cos(gamma.tensor) - torch.sin(alpha.tensor)*torch.sin(gamma.tensor))-x2.tensor)
        eqn6 = (y.tensor - h*torch.sin(alpha.tensor)*torch.cos(beta.tensor) + k2*w*(torch.cos(alpha.tensor)*torch.sin(beta.tensor)*torch.sin(gamma.tensor) + torch.cos(alpha.tensor)*torch.cos(gamma.tensor)) - l2*(torch.sin(alpha.tensor)*torch.sin(beta.tensor)*torch.cos(gamma.tensor) - torch.cos(alpha.tensor)*torch.sin(gamma.tensor))-y2.tensor)
        eqn7 = (z.tensor + h*torch.sin(beta.tensor) + k2*w*torch.cos(beta.tensor)*torch.sin(gamma.tensor) - l2*torch.cos(beta.tensor)*torch.cos(gamma.tensor) - z2.tensor)
        eqn8 = (z2.tensor - lamda*torch.exp((-a*x2.tensor**2)+(-a*y2.tensor**2))) 
        # eqn8 = (z2.tensor - (0.5*torch.sin(x2.tensor) + 0.5*torch.cos(y2.tensor)))
        # eqn8 = z2.tensor - (x2.tensor+y2.tensor) 
        
        eqn9 = (x.tensor - h*torch.cos(alpha.tensor)*torch.cos(beta.tensor) + k3*w*(torch.cos(alpha.tensor)*torch.sin(beta.tensor)*torch.sin(gamma.tensor) - torch.sin(alpha.tensor)*torch.cos(gamma.tensor)) - l3*(torch.cos(alpha.tensor)*torch.sin(beta.tensor)*torch.cos(gamma.tensor) - torch.sin(alpha.tensor)*torch.sin(gamma.tensor))-x3.tensor)
        eqn10 = (y.tensor - h*torch.sin(alpha.tensor)*torch.cos(beta.tensor) + k3*w*(torch.cos(alpha.tensor)*torch.sin(beta.tensor)*torch.sin(gamma.tensor) + torch.cos(alpha.tensor)*torch.cos(gamma.tensor)) - l3*(torch.sin(alpha.tensor)*torch.sin(beta.tensor)*torch.cos(gamma.tensor) - torch.cos(alpha.tensor)*torch.sin(gamma.tensor))-y3.tensor)
        eqn11 = (z.tensor + h*torch.sin(beta.tensor) + k3*w*torch.cos(beta.tensor)*torch.sin(gamma.tensor) - l3*torch.cos(beta.tensor)*torch.cos(gamma.tensor) - z3.tensor)
        eqn12 = (z3.tensor - lamda*torch.exp((-a*x3.tensor**2)+(-a*y3.tensor**2))) 
        # eqn12 = (z3.tensor - (0.5*torch.sin(x3.tensor) + 0.5*torch.cos(y3.tensor)))
        # eqn12 = z3.tensor - (x3.tensor+y3.tensor) 
        
        eqn13 = (x.tensor + h*torch.cos(alpha.tensor)*torch.cos(beta.tensor) + k4*w*(torch.cos(alpha.tensor)*torch.sin(beta.tensor)*torch.sin(gamma.tensor) - torch.sin(alpha.tensor)*torch.cos(gamma.tensor)) - l4*(torch.cos(alpha.tensor)*torch.sin(beta.tensor)*torch.cos(gamma.tensor) - torch.sin(alpha.tensor)*torch.sin(gamma.tensor))-x4.tensor)
        eqn14 = (y.tensor + h*torch.sin(alpha.tensor)*torch.cos(beta.tensor) + k4*w*(torch.cos(alpha.tensor)*torch.sin(beta.tensor)*torch.sin(gamma.tensor) + torch.cos(alpha.tensor)*torch.cos(gamma.tensor)) - l4*(torch.sin(alpha.tensor)*torch.sin(beta.tensor)*torch.cos(gamma.tensor) - torch.cos(alpha.tensor)*torch.sin(gamma.tensor))-y4.tensor)
        eqn15 = (z.tensor - h*torch.sin(beta.tensor) + k4*w*torch.cos(beta.tensor)*torch.sin(gamma.tensor) - l4*torch.cos(beta.tensor)*torch.cos(gamma.tensor) - z4.tensor)
        eqn16 = (z4.tensor - lamda*torch.exp((-a*x4.tensor**2)+(-a*y4.tensor**2))) 
        # eqn16 = (z4.tensor - (0.5*torch.sin(x4.tensor) + 0.5*torch.cos(y4.tensor)))
        eqn17 = torch.min(F.relu(gamma.tensor),torch.tensor([np.deg2rad(90)],device=self.device))
        eqn18 = torch.min(F.relu(beta.tensor),torch.tensor([np.deg2rad(90)],device=self.device))
        # eqn19 = F.relu(z3.tensor)
        # eqn20 = F.relu(z4.tensor)

        err = torch.cat((
            eqn1, eqn2, eqn3, eqn4, eqn5, eqn6, eqn7, eqn8,
            eqn9, eqn10, eqn11, eqn12, eqn13, eqn14, eqn15, eqn16, eqn17, eqn18),dim=1)
        # err = (
        #     eqn1**2 + eqn2**2 + eqn3**2 + eqn4**2 + eqn5**2 + eqn6**2 + eqn7**2 + eqn8**2 +
        #     eqn9**2 + eqn10**2 + eqn11**2 + eqn12**2 + eqn13**2 + eqn14**2 + eqn15**2 + eqn16**2 + eqn17**2 + eqn18**2)
        return err
    
    def bspline(self, c_arr, t_arr=None, n=30, degree=3):
        sample_device = c_arr.device
        sample_dtype = c_arr.dtype
        cv = c_arr.cpu().numpy()
        if(t_arr is None):
            t_arr = np.linspace(0, cv.shape[0], cv.shape[0])
        # else:
        #     t_arr = t_arr.cpu().numpy()
        spl = si.splrep(t_arr, cv, k=degree, s=0.0)
        #spl = BSpline(t, c, k, extrapolate=False)
        xx = np.linspace(0, n, n)
        # print(xx)
        # quit()
        samples = si.splev(xx, spl, ext=3)
        samples = torch.as_tensor(samples, device=sample_device, dtype=sample_dtype)
        return samples
    
    def scale_controls(self, act_seq):
        return torch.max(torch.min(act_seq, self.max_ctrl),self.min_ctrl)

    def sample_controls(self):
        uniform_halton_samples = torch.tensor(self.sequencer.get(self.sample_shape)) # samples N control points
        erfinv = torch.erfinv(2 * uniform_halton_samples - 1)
        knot_points = torch.sqrt(torch.tensor([2.0])) * erfinv
        # print(knot_points.shape)
        knot_samples = knot_points.view(self.sample_shape, self.d_action, self.n_knots)
        # print(knot_samples.shape)
        self.samples = torch.zeros((self.sample_shape, self.horizon, self.d_action))
        # print(self.samples.shape)
        for i in range(self.sample_shape):
            for j in range(self.d_action):
                self.samples[i,:,j] = self.bspline(knot_samples[i,j,:],n = self.horizon, \
                                                            degree = self.bspline_degree)
        delta = self.samples
        # z_seq = torch.zeros(1,self.horizon,self.d_action)
        # delta = torch.cat((delta,z_seq),dim=0)
        scaled_delta = torch.matmul(delta, self.full_scale_tril.float()).view(delta.shape[0],
                                                                    self.horizon,
                                                                    self.d_action)    
        act_seq = self.mean_action.unsqueeze(0) + scaled_delta
        act_seq = self.scale_controls(act_seq)
        append_acts = self.best_action.unsqueeze(0)

        act_seq = torch.cat((act_seq, append_acts), dim=0)
        self.controls_N = act_seq

    def unicycle_model(self, state, controls):
        a = torch.tensor([
            [torch.cos(state[2]), 0],
            [torch.sin(state[2]), 0],
            [0, 1]
            ],dtype=torch.float32)
        state = state + a@controls.float()*self.dt
        return state

    def surface_eq(self, x, y):
        lamda = 2
        a = 0.05
        return lamda*np.exp((-a*x**2)+(-a*y**2))
        # return 0.5 * np.sin(x) + 0.5 * np.cos(y)

    def equations(self, opt_vars, args):
        k1 = (2.5-1)/np.abs(2.5-1) # for i=1
        k2 = (2.5-2)/np.abs(2.5-2) # for i=2
        k3 = (2.5-3)/np.abs(2.5-3) # for i=3
        k4 = (2.5-4)/np.abs(2.5-4) # for i=4
        h = 0.901 #breadth
        w = 0.7406 #width
        l1,l2,l3,l4 = 0.1651, 0.1651, 0.1651, 0.1651
        lamda = 2
        a = 0.05

        x, y, alpha = args[0], args[1], args[2]

        z = opt_vars[0]
        beta = opt_vars[1]
        gamma = opt_vars[2]

        x1 = opt_vars[3]
        y1 = opt_vars[4]
        z1 = opt_vars[5]

        x2 = opt_vars[6]
        y2 = opt_vars[7]
        z2 = opt_vars[8]

        x3 = opt_vars[9]
        y3 = opt_vars[10]
        z3 = opt_vars[11]

        x4 = opt_vars[12]
        y4 = opt_vars[13]
        z4 = opt_vars[14]

        eqn1 = x + h*np.cos(alpha)*np.cos(beta) + k1*w*(np.cos(alpha)*np.sin(beta)*np.sin(gamma) - np.sin(alpha)*np.cos(gamma)) - l1*(np.cos(alpha)*np.sin(beta)*np.cos(gamma) - np.sin(alpha)*np.sin(gamma))-x1 
        eqn2 = y + h*np.sin(alpha)*np.cos(beta) + k1*w*(np.cos(alpha)*np.sin(beta)*np.sin(gamma) + np.cos(alpha)*np.cos(gamma)) - l1*(np.sin(alpha)*np.sin(beta)*np.cos(gamma) - np.cos(alpha)*np.sin(gamma))-y1 
        eqn3 = z - h*np.sin(beta) + k1*w*np.cos(beta)*np.sin(gamma) - l1*np.cos(beta)*np.cos(gamma) - z1 
        eqn4 = z1 - lamda*np.exp((-a*x1**2)+(-a*y1**2)) #(0.5*np.sin(x1) + 0.5*np.cos(y1)) 

        eqn5 = x - h*np.cos(alpha)*np.cos(beta) + k2*w*(np.cos(alpha)*np.sin(beta)*np.sin(gamma) - np.sin(alpha)*np.cos(gamma)) - l2*(np.cos(alpha)*np.sin(beta)*np.cos(gamma) - np.sin(alpha)*np.sin(gamma))-x2 
        eqn6 = y - h*np.sin(alpha)*np.cos(beta) + k2*w*(np.cos(alpha)*np.sin(beta)*np.sin(gamma) + np.cos(alpha)*np.cos(gamma)) - l2*(np.sin(alpha)*np.sin(beta)*np.cos(gamma) - np.cos(alpha)*np.sin(gamma))-y2 
        eqn7 = z + h*np.sin(beta) + k2*w*np.cos(beta)*np.sin(gamma) - l2*np.cos(beta)*np.cos(gamma) - z2 
        eqn8 = z2 - lamda*np.exp((-a*x2**2)+(-a*y2**2)) #(0.5*np.sin(x2) + 0.5*np.cos(y2)) 

        eqn9 = x - h*np.cos(alpha)*np.cos(beta) + k3*w*(np.cos(alpha)*np.sin(beta)*np.sin(gamma) - np.sin(alpha)*np.cos(gamma)) - l3*(np.cos(alpha)*np.sin(beta)*np.cos(gamma) - np.sin(alpha)*np.sin(gamma))-x3 
        eqn10 = y - h*np.sin(alpha)*np.cos(beta) + k3*w*(np.cos(alpha)*np.sin(beta)*np.sin(gamma) + np.cos(alpha)*np.cos(gamma)) - l3*(np.sin(alpha)*np.sin(beta)*np.cos(gamma) - np.cos(alpha)*np.sin(gamma))-y3 
        eqn11 = z + h*np.sin(beta) + k3*w*np.cos(beta)*np.sin(gamma) - l3*np.cos(beta)*np.cos(gamma) - z3
        eqn12 = z3 - lamda*np.exp((-a*x3**2)+(-a*y3**2)) #(0.5*np.sin(x3) + 0.5*np.cos(y3)) 

        eqn13 = x + h*np.cos(alpha)*np.cos(beta) + k4*w*(np.cos(alpha)*np.sin(beta)*np.sin(gamma) - np.sin(alpha)*np.cos(gamma)) - l4*(np.cos(alpha)*np.sin(beta)*np.cos(gamma) - np.sin(alpha)*np.sin(gamma))-x4 
        eqn14 = y + h*np.sin(alpha)*np.cos(beta) + k4*w*(np.cos(alpha)*np.sin(beta)*np.sin(gamma) + np.cos(alpha)*np.cos(gamma)) - l4*(np.sin(alpha)*np.sin(beta)*np.cos(gamma) - np.cos(alpha)*np.sin(gamma))-y4 
        eqn15 = z - h*np.sin(beta) + k4*w*np.cos(beta)*np.sin(gamma) - l4*np.cos(beta)*np.cos(gamma) - z4
        eqn16 = z4 - lamda*np.exp((-a*x4**2)+(-a*y4**2)) #(0.5*np.sin(x4) + 0.5*np.cos(y4)) 

        return [eqn1, eqn2, eqn3, eqn4, eqn5, eqn6, eqn7, eqn8, eqn9, eqn10, eqn11, eqn12, eqn13, eqn14, eqn15, eqn16]


    def rollout(self):
        diag_dt = self.dt*torch.ones(self.horizon, self.horizon)
        diag_dt = torch.tril(diag_dt)
        for i in range(self.controls_N.shape[0]):
            # t1 = time.time()
            self.traj_N[i,0,:3] = self.c_state.view(3)
            for j in range(1,self.traj_N.shape[1]):
                self.traj_N[i,j,:3] = self.unicycle_model(self.traj_N[i,j-1,:3], self.controls_N[i,j-1,:])
            # print(self.traj_N[i,:,:3])
            # quit()
            # # self.traj_N[i,0,:3] = self.c_state.view(3)
            # v = self.controls_N[i,:,0].view(-1,1)
            # w = self.controls_N[i,:,1].view(-1,1)
            # w_dt = diag_dt@w.float()
            # theta_0 = self.traj_N[i,0,2]*torch.ones(self.horizon,1)
            # x_0 = self.traj_N[i,0,0]*torch.ones(self.horizon,1)
            # y_0 = self.traj_N[i,0,1]*torch.ones(self.horizon,1)
            # theta_new = theta_0 + w_dt
            # c_theta = torch.cos(theta_new)
            # s_theta = torch.sin(theta_new)
            # v_cos_dt = (c_theta.squeeze(1)*diag_dt)@v.float()
            # v_sin_dt = (s_theta.squeeze(1)*diag_dt)@v.float()
            # x_new = x_0 + v_cos_dt
            # y_new = y_0 + v_sin_dt
            # self.traj_N[i,1:,:3] = torch.hstack((x_new, y_new, theta_new))

        batched_traj_N = self.traj_N.reshape(self.traj_N.shape[0]*self.traj_N.shape[1],self.traj_N.shape[2])
        x_ten = batched_traj_N[:,0]
        x_ten = x_ten.reshape(-1,1)
        y_ten = batched_traj_N[:,1]
        y_ten = y_ten.reshape(-1,1)
        theta_ten = batched_traj_N[:,2]
        theta_ten = theta_ten.reshape(-1,1)
        theseus_inputs = {
            "x": x_ten.to(self.device), #torch.tensor([[-6],[5]], dtype=torch.float32),
            "y": y_ten.to(self.device), #torch.tensor([[-6],[5]], dtype=torch.float32),
            "theta": theta_ten.to(self.device), #torch.tensor([[1.5707964],[np.deg2rad(45)]], dtype=torch.float32),
        }
        tic = time.time()
        with torch.no_grad():
            updated_inputs, info  = self.theseus_optim.forward(
                theseus_inputs, optimizer_kwargs={"track_best_solution": True, "verbose":False})
        toc = time.time()
        # print("time taken: ", toc-tic)

        th_sol = []
        z_th,beta_th,gamma_th, x1_th,y1_th,z1_th, x2_th,y2_th,z2_th, x3_th,y3_th,z3_th, x4_th,y4_th,z4_th = info.best_solution.values()
        for i in range(len(z_th)):
            th_sol.append([z_th[i].item(),beta_th[i].item(),gamma_th[i].item(),
                        x1_th[i].item(),y1_th[i].item(),z1_th[i].item(),
                        x2_th[i].item(),y2_th[i].item(),z2_th[i].item(),
                        x3_th[i].item(),y3_th[i].item(),z3_th[i].item(),
                        x4_th[i].item(),y4_th[i].item(),z4_th[i].item()])
        th_sol = torch.tensor(th_sol)

        batched_traj_N[:, 3:] = th_sol
        self.traj_N = batched_traj_N.reshape(self.traj_N.shape[0], self.traj_N.shape[1], self.traj_N.shape[2])

        # dist_to_best_cost_N = self.dist_to_best_cost()
        # top_values, top_idx = torch.topk(dist_to_best_cost_N, int(self.num_particles*0.99), largest=False, sorted=True)
        # closest_to_best_controls = torch.index_select(self.controls_N, 0, top_idx)
        # closest_to_best_traj = torch.index_select(self.traj_N, 0, top_idx) # x,y,theta, z,beta,gamma, x1,y1,z1, x2,y2,z2, x3,y3,z3, x4,y4,z4
        ang_vel_cost = self.ang_vel_cost(self.controls_N)
        vel_cost = self.vel_cost(self.controls_N)
        acc_cost_N = self.acc_cost(self.controls_N, [self.amin, self.amax])
        ang_acc_cost_N = self.ang_acc_cost(self.controls_N)
        goal_region_cost_N = self.goal_region_cost(self.traj_N, self.controls_N, radius = 1.0)
        beta_cost_N = self.beta_cost(self.traj_N)
        gamma_cost_N = self.gamma_cost(self.traj_N)

        # acc_cost_N = (acc_cost_N-acc_cost_N.min())/(acc_cost_N.max()-acc_cost_N.min())
        # ang_acc_cost_N = (ang_acc_cost_N-ang_acc_cost_N.min())/(ang_acc_cost_N.max()-ang_acc_cost_N.min())
        # ang_vel_cost = (ang_vel_cost-ang_vel_cost.min())/(ang_vel_cost.max()-ang_vel_cost.min())
        # vel_cost = (vel_cost-vel_cost.min())/(vel_cost.max()-vel_cost.min())
        # goal_region_cost_N = (goal_region_cost_N-goal_region_cost_N.min())/(goal_region_cost_N.max()-goal_region_cost_N.min())
        # beta_cost_N = (beta_cost_N-beta_cost_N.min())/(beta_cost_N.max()-beta_cost_N.min())
        # gamma_cost_N = (gamma_cost_N-gamma_cost_N.min())/(gamma_cost_N.max()-gamma_cost_N.min())

        # self.total_cost_N =  1*goal_region_cost_N + 2*beta_cost_N + 1*gamma_cost_N#0*acc_cost_N + 0*ang_acc_cost_N + 0*ang_vel_cost + 0*vel_cost +\
            #   1*goal_region_cost_N + 0*beta_cost_N + 0*gamma_cost_N
        self.total_cost_N =  1*goal_region_cost_N + 2*beta_cost_N + 1*gamma_cost_N#0*acc_cost_N + 0*ang_acc_cost_N + 0*ang_vel_cost + 0*vel_cost +\

        
        top_values, top_idx = torch.topk(self.total_cost_N, self.top_K, largest=False, sorted=True)
        # print(top_idx)
        # print(self.controls_N.shape)
        self.top_trajs = torch.index_select(self.traj_N, 0, top_idx)
        top_controls = torch.index_select(self.controls_N, 0, top_idx)
        # self.goal_region_cost(self.top_trajs, top_controls, radius = 1.0)
        self.best_action = copy.deepcopy(top_controls[0,:,:])
        top_cost = torch.index_select(self.total_cost_N, 0, top_idx)
        top_w = self._exp_util(top_cost)
        return top_w, top_controls

    
    def goal_region_cost(self, traj, controls, radius = 3.5):
        goal_region_cost_N = torch.zeros((traj.shape[0]))

        for i in range(traj.shape[0]):
            goal_region_cost_N[i] += torch.linalg.norm(traj[i, -1,:3] - self.g_state[:3])             

        return goal_region_cost_N

    def acc_cost(self, controls, acc_bound = [-3, 3]):
        dt = self.dt
        acc_cost_N = torch.zeros((controls.shape[0], controls.shape[1]))
        for i in range(controls.shape[1]-1):
            acc_cost_N[:,i] = (controls[:,i+1,0] - controls[:,i,0])/dt

        acc_cost_N[(acc_cost_N>=acc_bound[0]) & (acc_cost_N<=acc_bound[1])] = 0
        return torch.sum(acc_cost_N,axis=1)

    def dist_to_best_cost(self):
        dist_to_best_cost_N = torch.zeros((self.controls_N.shape[0]))
        for i in range(self.controls_N.shape[0]):
            dist_to_best_cost_N[i] = torch.linalg.norm(self.controls_N[i,:,:] -  self.best_action)
        return dist_to_best_cost_N

    def ang_acc_cost(self, controls, ang_acc_bound = [-0.1, 0.1]):
        dt = self.dt
        ang_acc_cost_N = torch.zeros((controls.shape[0], controls.shape[1]))
        for i in range(controls.shape[1]-1):
            ang_acc_cost_N[:,i] = (controls[:,i+1,1] - controls[:,i,1])/dt
        ang_acc_cost_N[(ang_acc_cost_N>=self.jmin) & (ang_acc_cost_N<=self.jmax)] = 0
        return torch.sum(ang_acc_cost_N,axis=1)
    
    def vel_cost(self, controls):
        dt = self.dt
        vel_cost_N = torch.zeros((controls.shape[0]))
        ang_vel_cost = 0
        # for i in range(controls.shape[1]):
        vel_cost_N[:] += controls[:,-1,0]**2
        return vel_cost_N

    def ang_vel_cost(self, controls):
        dt = self.dt
        ang_vel_cost_N = torch.zeros((controls.shape[0]))
        ang_vel_cost = 0
        for i in range(controls.shape[1]):
            ang_vel_cost_N[:] += controls[:,i,1]**2
        return ang_vel_cost_N
    
    def beta_cost(self, traj):
        beta_cost_N = torch.zeros((traj.shape[0]))
        for i in range(1,traj.shape[1]):
            beta_cost_N[:] += traj[:,i,4]**2 #(traj[:,i,4] - traj[:,i-1,4])**2 #+ traj[:,i,4]**2
        return beta_cost_N
    
    def gamma_cost(self, traj):
        gamma_cost_N = torch.zeros((traj.shape[0]))
        for i in range(1, traj.shape[1]):
            gamma_cost_N[:] += traj[:,i,5]**2 #(traj[:,i,5] - traj[:,i-1,5])**2 #+ traj[:,i,5]**2
        return gamma_cost_N


    def collision_cost(self, traj):
        radius = 1
        collision_cost_N = torch.zeros((traj.shape[0]))
        for n in range(0, traj.shape[0]):
            n_sum = 0
            for t in range(0, traj.shape[1]):
                for o in self.obstacles:
                    dist = torch.linalg.norm(traj[n,t,:2]- o[:2])
                    if(dist <=2*radius):
                        n_sum += 5000
                collision_cost_N[n] += n_sum
        return collision_cost_N

    
    def _exp_util(self, costs):
        """
            Calculate weights using exponential utility
        """
        beta = 0.99
        # cost_seq = costs  # discounted cost sequence
        # cost_seq = torch.fliplr(torch.cumsum(torch.fliplr(cost_seq), axis=-1))  # cost to go (but scaled by [1 , gamma, gamma*2 and so on])
        # cost_seq /= self.gamma_seq  # un-scale it to get true discounted cost to go
        traj_costs = costs

        # traj_costs = torch.sum(traj_costs,1)
        # self.total_cost_N = traj_costs
        # #calculate soft-max
        w = torch.softmax((-1.0/beta) * traj_costs, dim=0)
        return w
    
    def update_distribution(self, top_w, top_controls):
        
        weighted_seq = top_w.to(self.device) * top_controls.to(self.device).T        
        sum_seq = torch.sum(weighted_seq.T, dim=0)

        new_mean = sum_seq
        self.mean_action = (1.0 - self.step_size_mean) * self.mean_action +\
            self.step_size_mean * new_mean
        
        delta = top_controls - self.mean_action.unsqueeze(0)

        # weighted_delta = top_w * (delta ** 2).T
        # # cov_update = torch.diag(torch.mean(torch.sum(weighted_delta.T, dim=0), dim=0))
        # cov_update = torch.mean(torch.sum(weighted_delta.T, dim=0), dim=0)
        # self.cov_action = (1.0 - self.step_size_cov) * self.cov_action +\
        #         self.step_size_cov * cov_update
    
    def plan_traj(self):
        t1 = time.time()
        # top_w, top_controls = self.get_cost()
        self.cov_action = self.init_cov_action
        # self.scale_tril = torch.sqrt(self.cov_action)
        # self.full_scale_tril = torch.diag(self.scale_tril)
        for i in range(2):
            self.scale_tril = torch.sqrt(self.cov_action)
            self.full_scale_tril = torch.diag(self.scale_tril)
            self.sample_controls()
            # print("Sample Controls: ", time.time() - t1)
            top_w, top_controls = self.rollout()
            # print("Rollout: ", time.time() - t1)
            t1 = time.time()
            self.update_distribution(top_w, top_controls)
            # print("Update_Distribution: ", time.time() - t1)
            # print("#######################")
        self.scale_tril = torch.sqrt(self.cov_action)
        self.full_scale_tril = torch.diag(self.scale_tril)
        self.sample_controls()
        top_w, top_controls = self.rollout()
        # self.mean_action[:-1,:] = self.mean_action[1:,:].clone()
        # self.mean_action[-1,:] = self.init_mean[-1,:].clone()

    def get_vel(self, u):
        v1 = self.vl
        w1 = self.wl
        v = torch.zeros(u.shape)
        for i in range(u.shape[1]):
            v[0,i] = v1 + u[0,i]*self.dt
            v1 = v[0,i]
            v[1,i] = w1 + u[0,i]*self.dt
            w1 = v[1,i]       
        return v

def terrain_equation(x, y):
    lamda = 2.5
    a = 0.05
    return lamda*np.exp((-a*x**2)+(-a*y**2))
    # return lambda*np.exp((-a*x**2)+(-a*y**2))
    return 0.5*np.sin(x) + 0.5*np.cos(y)

def xz(phi,r=0.1651):
    return r*np.cos(phi), r*np.sin(phi)

def plot_vehicle(contact_points, R_arr, g_state, col='red'):
    '''
    N = number of poses in the trajectory
    contact_points is of shape N*4,3. # contains x_i, y_i, z_i
    R_arr is of shape N*4,3. # contains alpha, beta, gamma
    '''
    os.makedirs('frames', exist_ok=True)
    save_name = 0
    phis=np.arange(0,6.28,0.01)
    x_wheel, z_wheel = xz(phis)
    y_wheel = np.zeros(x_wheel.shape)
    x_wheel = x_wheel.reshape(1,x_wheel.shape[0])
    y_wheel = y_wheel.reshape(1,y_wheel.shape[0])
    z_wheel = z_wheel.reshape(1,z_wheel.shape[0])
    # print(contact_points.shape[0])
    wheel_coord = np.concatenate((x_wheel, y_wheel, z_wheel),0)
    p = np.array([[0],[0],[0.25]])
    
    for i in range(0,contact_points.shape[0],4): 
        fig = plt.figure()
        fig = plt.figure(figsize=(20, 16))
        ax = fig.add_subplot(111, projection='3d')
        plot_terrain(ax)
        ax.plot(g_state[0].cpu().item(), g_state[1].cpu().item(), terrain_equation(g_state[0].cpu().item(), g_state[1].cpu().item()), 'r.', markersize=20, zorder=4)
        
        x_c = contact_points[i:i+4,0]
        y_c = contact_points[i:i+4,1]
        z_c = contact_points[i:i+4,2]
        alpha, beta, gamma = R_arr[i,0], R_arr[i,1], R_arr[i,2]
        R_x = np.array([
            [1,0,0],
            [0, np.cos(gamma), -np.sin(gamma)],
            [0, np.sin(gamma), np.cos(gamma)],
        ])
        R_y = np.array([
            [np.cos(beta), 0, np.sin(beta)],
            [0, 1, 0],
            [-np.sin(beta), 0, np.cos(beta)],
        ])
        R_z = np.array([
            [np.cos(alpha), -np.sin(alpha), 0],
            [np.sin(alpha), np.cos(alpha), 0],
            [0, 0, 1],
        ])
        R = R_z@R_y@R_x
        new_wheel = R@wheel_coord # orienting the wheels by multiplying with the Rotation matrix
        elv = []
        for j in range(x_c.shape[0]):
            elv.append(R@p+np.array([[x_c[j]], [y_c[j]], [z_c[j]]])) # Elevated vehicle surface points
        elv = np.array(elv).reshape(-1,3)
        
        # Plotting the oriented wheels
        ax.plot(contact_points[i+0, 0]+new_wheel[0,:], contact_points[i+0, 1]+new_wheel[1,:], contact_points[i+0, 2]+new_wheel[2,:], 'k', markersize=20, zorder=4)
        ax.plot(contact_points[i+1, 0]+new_wheel[0,:], contact_points[i+1, 1]+new_wheel[1,:], contact_points[i+1, 2]+new_wheel[2,:], 'k', markersize=20, zorder=4)
        ax.plot(contact_points[i+2, 0]+new_wheel[0,:], contact_points[i+2, 1]+new_wheel[1,:], contact_points[i+2, 2]+new_wheel[2,:], 'k', markersize=20, zorder=4)
        ax.plot(contact_points[i+3, 0]+new_wheel[0,:], contact_points[i+3, 1]+new_wheel[1,:], contact_points[i+3, 2]+new_wheel[2,:], 'k', markersize=20, zorder=4)
        
        # Plotting the links connecting the body to the wheel
        for j in range(len(elv)):
            ax.plot([x_c[j],elv[j,0]], [y_c[j], elv[j,1]], [z_c[j], elv[j,2]], 'k', markersize=20, zorder=4)

        # Plotting the elevated surface of the vehicle
        ax.plot(elv[0:2,0], elv[0:2,1], elv[0:2,2], col, markersize=20, zorder=4, label='w1')
        ax.plot(elv[1:3,0], elv[1:3,1], elv[1:3,2], col, markersize=20, zorder=4, label='w1')
        ax.plot(elv[2:4,0], elv[2:4,1], elv[2:4,2], col, markersize=20, zorder=4, label='w1')
        ax.plot([elv[3,0], elv[0,0]], [elv[3,1], elv[0,1]], [elv[3,2], elv[0,2]], col, markersize=10, zorder=4, label='w1')
        
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_zlabel('Z-axis')
        ax.set_xlim(-2, 4)
        ax.set_ylim(-2, 4)
        ax.set_zlim(-2, 4)
        # # for ii in range(0,360,10):
        ax.view_init(elev=17, azim=300)
        if(i>0):
            save_name += 1 
            print(save_name)
        plt.savefig("frames/"+str(save_name)+".png",bbox_inches='tight',dpi=200)
        plt.close('all')
        
        # plt.legend()
    # plt.show()

def plot_terrain(ax):
    x = np.linspace(-10, 10, 100)
    y = np.linspace(-10, 10, 100)
    X, Y = np.meshgrid(x, y)
    Z = terrain_equation(X, Y)
    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.99, zorder=1, linewidth=0.1)

def set_plot_params(ax):#,i,c_state):
    os.makedirs('frames', exist_ok=True)
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    ax.set_xlim(-2, 4)
    ax.set_ylim(-2, 4)
    ax.set_zlim(-2, 4)
    # for ii in range(0,360,10):
    ax.view_init(elev=17, azim=300)
    plt.savefig("frames/"+str(i)+".png",bbox_inches='tight',dpi=500)
    # plt.legend()
    plt.show()

if __name__ == '__main__':
    dtype = torch.float32
    c_state = torch.tensor([1.5, -6, np.deg2rad(90)], dtype=dtype).view(3,1) # sample his randomly
    g_state = torch.tensor([0, 5, np.deg2rad(90)], dtype=dtype).view(3,1) # sample this randomly 
    # make sure torch.norm(c_state-g_state) >10

    vl = 5
    wl = 0
    exec_traj = []

    sampler = Goal_Sampler(c_state, g_state, vl, wl, obstacles=[])
    t1 = time.time()
    sampler.plan_traj()
    t2 = time.time()
    # print(t2-t1)
    top_traj = sampler.top_trajs[0,:,:] # contains the best trajectory. To be saved # x,y,theta, z_th,beta_th,gamma_th, x1_th,y1_th,z1_th, x2_th,y2_th,z2_th, x3_th,y3_th,z3_th, x4_th,y4_th,z4_th
    top_controls = sampler.best_action # contains the best controls. To be saved
    # plt.plot(top_traj[:,4],'r',label='roll')
    # plt.plot(top_traj[:,5],'b',label='pitch')
    # plt.legend()
    # plt.title("With Roll smoothness cost")
    # fig = plt.figure()
    # fig = plt.figure(figsize=(20, 16))
    # ax = fig.add_subplot(111, projection='3d')
    # plot_terrain(ax)
    # ax.set_xlabel('X-axis')
    # ax.set_ylabel('Y-axis')
    # ax.set_zlabel('Z-axis')
    # ax.set_xlim(-2, 4)
    # ax.set_ylim(-2, 4)
    # ax.set_zlim(-2, 4)
    # plt.show()
    # quit()
    # print(sampler.top_trajs.shape)
    # print(top_traj.shape)
    # print(top_controls.shape)
    # exec_traj.append(sampler.top_trajs[0,0,:])
    # exec_traj.append(sampler.top_trajs[0,1,:])
    # sampler.c_state = sampler.top_trajs[0,1,:3]
    # k = 0
    # print(torch.linalg.norm(sampler.c_state-sampler.g_state))
    # best_dist = 999
    # while torch.sqrt(torch.linalg.norm(sampler.c_state-sampler.g_state))>1 and k<=100:
    best_dist = 9999999
    lim_dist = 5
    for k in range(200):
        dist = torch.sqrt(torch.linalg.norm(sampler.c_state-sampler.g_state))
        print(dist)
        if(dist>best_dist and lim_dist>0):
            lim_dist -= 1
        if lim_dist <=0:
            break
        best_dist=copy.deepcopy(dist)
        print(k, best_dist)
        sampler.plan_traj()
        top_traj = sampler.top_trajs[0,:,:]
        exec_traj.append(sampler.top_trajs[0,1,:])
        sampler.c_state = sampler.top_trajs[0,1,:3]
        k+=1
    # exec_traj = top_traj #[0,:].reshape(1,-1)
    contact_points = []
    R_arr = []
    g_state = torch.tensor([exec_traj[-1][0], exec_traj[-1][1], np.deg2rad(90)], dtype=dtype).view(3,1) # sample this randomly 

    for j in range(len(exec_traj)):
        arr = []
        R_arr.append([exec_traj[j][2].cpu().numpy().item(), exec_traj[j][4], exec_traj[j][5]])
        R_arr.append([exec_traj[j][2].cpu().numpy().item(), exec_traj[j][4], exec_traj[j][5]])
        R_arr.append([exec_traj[j][2].cpu().numpy().item(), exec_traj[j][4], exec_traj[j][5]])
        R_arr.append([exec_traj[j][2].cpu().numpy().item(), exec_traj[j][4], exec_traj[j][5]])
        # print(j, exec_traj[j][:2])
    
        for i in range(6, 18):
            arr.append(exec_traj[j][i])
            if((i+1)%3 == 0):
                contact_points.append(arr)
                arr = []

    contact_points_sci = np.array(contact_points)
    R_arr_sci = np.array(R_arr)
    # print(contact_points_sci.shape)
    # print(R_arr_sci.shape)
    # quit()
    # fig = plt.figure()
    # fig = plt.figure(figsize=(20, 16))
    # ax = fig.add_subplot(111, projection='3d')
    # plot_terrain(ax)
    
    plot_vehicle(contact_points_sci, R_arr_sci, g_state, col='red')
    # plot_vehicle(contact_points_th, R_arr_th, ax, col='blue')
    
    # print(g_state[0].cpu().item(), g_state[1].cpu().item())
    
   
    # for i in range(sampler.traj_N.shape[0]):
    #     for j in range(sampler.traj_N.shape[1]):
    #         plt.plot(sampler.traj_N[i,j,0].numpy(), sampler.traj_N[i,j,1].numpy(), sampler.traj_N[i,j,3].numpy(), 'b.', markersize=2, zorder=4)
    # for i in range(sampler.top_trajs.shape[0]):
    #     for j in range(sampler.top_trajs.shape[1]):
    #         plt.plot(sampler.top_trajs[i,j,0].numpy(), sampler.top_trajs[i,j,1].numpy(), sampler.top_trajs[i,j,3].numpy(), 'r.', markersize=5, zorder=4)
    # for j in range(sampler.top_trajs.shape[1]):
    #     plt.plot(sampler.top_trajs[0,j,0].numpy(), sampler.top_trajs[0,j,1].numpy(), sampler.top_trajs[0,j,3].numpy(), 'r.', markersize=10, zorder=4)
    # set_plot_params(ax)