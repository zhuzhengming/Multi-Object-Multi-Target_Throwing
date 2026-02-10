""""
generate throwing trajectory by querying BRT and hedgehog
input: box_position
output: target[q, q_dot]
"""
import sys
import os

from lxml import etree

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, "../"))
sys.path.append(os.path.join(current_dir, "../../.."))
from hedgehog import VelocityHedgehog
import time
import math
import numpy as np
import itertools
from ruckig import InputParameter, Ruckig, Trajectory, Result
from utils.config_loader import get_param
import mujoco
from scipy.spatial import KDTree
import glfw
import gc
import pinocchio as pin


class TrajectoryGenerator:
    def __init__(self, q_ul, q_ll, hedgehog_path, brt_path, robot_path, box_position=None, model_exist=False, q0=None):
        self.q_ul = q_ul
        self.q_ll = q_ll
        self.q0_dot = np.zeros(7)
        self.hedgehog_path = hedgehog_path
        self.brt_path = brt_path
        self.gravity = -9.81
        self.GAMMA_TOLERANCE = 0.2/180.0*np.pi
        self.Z_TOLERANCE = 0.01
        self.box_position = box_position
        self.freq = 1000
        self.off_data_path = os.path.join(current_dir, "../../../training_data")

        if q0 is None:
            self.q0 = np.array([-1.5783, 0.1498, 0.1635, -0.7926, -0.0098, 0.6, 1.2881])
        else:
            self.q0 = q0


        # mujoco similator
        if robot_path is None:
            self.robot_path = os.path.join(current_dir, '../description/iiwa7_allegro_throwing.xml')
        else:
            self.robot_path = robot_path

        q_dot_max = np.array([1.71, 1.74, 1.745, 2.269, 2.443, 3.142, 3.142])
        q_dot_min = -q_dot_max

        self.robot = VelocityHedgehog(self.q_ll, self.q_ul, q_dot_min, q_dot_max, robot_path, model_exist=model_exist)
        
        # pinocchio model for torque/energy calculation
        urdf_path = os.path.join(current_dir, '../description/iiwa_description/urdf/iiwa7_lasa.urdf')
        self.pin_model = pin.buildModelFromUrdf(urdf_path)
        self.pin_data = self.pin_model.createData()

        self.load_params()
        self.load_data()

    def load_params(self):
        self.max_velocity = np.array(get_param('/max_velocity')) 
        # self.max_acceleration = np.array([15, 7.5, 10, 12.5, 15, 20, 20])
        # self.max_jerk = np.array([7500, 3750, 5000, 6250, 7500, 10000, 10000])
        self.max_acceleration = np.array(get_param('/max_acceleration'))
        self.max_jerk = np.array(get_param('/max_jerk'))
        self.MARGIN_VELOCITY = get_param('/MARGIN_VELOCITY')
        self.MARGIN_ACCELERATION = get_param('/MARGIN_ACCELERATION')
        self.MARGIN_JERK = get_param('/MARGIN_JERK')

        self.thres_dis = get_param('/thres_dis')
        self.thres_v = get_param('/thres_v')
        self.thres_r_ratio = get_param('/thres_r_ratio')
        self.thres_height = get_param('/thres_height')
        self.thres_phi_ratio = get_param('/thres_phi_ratio')

    def load_data(self):
        # load hedgehog data
        self.robot_zs = np.load(self.hedgehog_path + '/robot_zs.npy')
        self.robot_dis = np.load(self.hedgehog_path + '/robot_diss.npy')
        self.robot_phis = np.load(self.hedgehog_path + '/robot_phis.npy')
        self.robot_gamma = np.load(self.hedgehog_path + '/robot_gammas.npy')
        self.zs_step = 0.05

        self.robot_phi_gamma_velos_naive_all = np.load(self.hedgehog_path + '/z_dis_phi_gamma_vel_max.npy')

        # different hedgehog data for different posture
        # posture1
        self.p1_robot_phi_gamma_velos_naive = self.robot_phi_gamma_velos_naive_all[0]
        self.p1_robot_phi_gamma_q_idxs_naive = np.load(self.hedgehog_path + '/q_idxs_posture1.npy')
        self.p1_ae = np.load(self.hedgehog_path + '/q_idx_ae_posture1.npy')
        self.p1_mesh = np.load(self.hedgehog_path + '/q_idx_qs_posture1.npy')

        # posture2
        self.p2_robot_phi_gamma_velos_naive = self.robot_phi_gamma_velos_naive_all[1]
        self.p2_robot_phi_gamma_q_idxs_naive = np.load(self.hedgehog_path + '/q_idxs_posture2.npy')
        self.p2_ae = np.load(self.hedgehog_path + '/q_idx_ae_posture2.npy')
        self.p2_mesh = np.load(self.hedgehog_path + '/q_idx_qs_posture2.npy')

        # load brt data
        self.brt_tensor = np.load(self.brt_path + '/brt_tensor.npy')
        self.brt_zs = np.load(self.brt_path + '/brt_zs.npy')


    def brt_robot_data_matching(self, posture, box_pos=None):
        """
        original point is the base of robot
        Given target position, find out initial guesses of (q, phi, x)
        :param box_position:
        :param thres_dis:
        :param thres_v:
        :param thres_r_ratio:
        :param thres_height:
        :param thres_phi_ratio:

        :return: candidates of q, phi, x
        """
        if posture == "posture1":
            ae = self.p1_ae
            mesh = self.p1_mesh
            robot_phi_gamma_velos_naive = self.p1_robot_phi_gamma_velos_naive
            robot_phi_gamma_q_idxs_naive = self.p1_robot_phi_gamma_q_idxs_naive
        elif posture == "posture2":
            ae = self.p2_ae
            mesh = self.p2_mesh
            robot_phi_gamma_velos_naive = self.p2_robot_phi_gamma_velos_naive
            robot_phi_gamma_q_idxs_naive = self.p2_robot_phi_gamma_q_idxs_naive

        box_pos = self.box_position if box_pos is None else box_pos
        z_target_to_base = box_pos[-1]
        AB = box_pos[:2]
        b = np.linalg.norm(AB)


        # align the z idx
        num_robot_zs = self.robot_zs.shape[0]
        num_brt_zs = self.brt_zs.shape[0]
        brt_z_min, brt_z_max = np.min(self.brt_zs), np.max(self.brt_zs)
        if z_target_to_base + brt_z_min > min(self.robot_zs):
            rzs_idx_start = round((z_target_to_base + brt_z_min) / self.zs_step)
            bzs_idx_start = 0
        else:
            rzs_idx_start = 0
            bzs_idx_start = -round((z_target_to_base + brt_z_min) / self.zs_step)
        if z_target_to_base + brt_z_max > max(self.robot_zs):
            rzs_idx_end = num_robot_zs - 1
            bzs_idx_end = num_brt_zs - 1 - round((z_target_to_base + brt_z_max - max(self.robot_zs)) / self.zs_step)
        else:
            rzs_idx_end = num_robot_zs - 1 + round((z_target_to_base + brt_z_max - max(self.robot_zs)) / self.zs_step)
            bzs_idx_end = num_brt_zs - 1
        assert bzs_idx_end - bzs_idx_start == rzs_idx_end - rzs_idx_start, \
            "bzs: {0}, {1}; rzs: {2}, {3}".format(bzs_idx_start, bzs_idx_end, rzs_idx_start, rzs_idx_end)
        z_num = bzs_idx_end - bzs_idx_start + 1
        if z_num == 0 or rzs_idx_end <= 0:
            return [], [], []

        # BRT-Tensor = {z, dis, phi, gamma, layers} -> [r, z, r_dot, z_dot]
        brt_tensor_slice = self.brt_tensor[bzs_idx_start:bzs_idx_end + 1, ...]

        # Fixed-base limitation
        # Robot tensor = [z, dis, phi, gamma, layers] -> [r, z, r_dot, z_dot, max_v]
        robot_tensor_v = np.expand_dims(robot_phi_gamma_velos_naive[rzs_idx_start: rzs_idx_end + 1, ...], axis=4)

        ################################### Selection ###############################################
        #############################################################################################
        # 1.distance < b
        b = np.linalg.norm(AB) # from target position
        robot_tensor_v = robot_tensor_v[:, np.where(self.robot_dis < b)[0], ...]

        # 2 calculate desired r
        # given [dis, phi, target_position] -> [r, z, r_dot, z_dot] -> [r, gamma]
        cos_phi = np.cos(self.robot_phis)
        sin_phi = np.sin(self.robot_phis)
        d_cosphi = self.robot_dis[self.robot_dis < b, np.newaxis] @ cos_phi[np.newaxis, :]
        d_sinphi = self.robot_dis[self.robot_dis < b, np.newaxis] @ sin_phi[np.newaxis, :]
        r = np.sqrt(b ** 2 - d_sinphi ** 2) - d_cosphi
        r_tensor = r[None, :, :, None, None] #[None, dis, phi, None, None]
        mask_r = abs(-brt_tensor_slice[:, :, :, :, :, 0] - r_tensor) < self.thres_dis

        # 3.choose these brt data which are close to r wrt thres_v
        validate = np.argwhere((robot_tensor_v -
                                self.thres_v - brt_tensor_slice[:, :, :, :, :, 4] > 0)  # velocity satisfy
                               * mask_r)

        q_indices = np.copy(validate[:, :4])
        q_indices[:, 0] += rzs_idx_start
        if np.any(q_indices >= robot_phi_gamma_q_idxs_naive.shape[0]):
            return [], [], []

        qids = robot_phi_gamma_q_idxs_naive[tuple(q_indices.T)].astype(int)
        q_candidates = mesh[qids, :]
        q_ae = ae[qids]
        phi_candidates = self.robot_phis[validate[:, 2]]
        x_candidates = brt_tensor_slice[:, 0, 0, :, :, :][tuple(np.r_['-1', validate[:, :1], validate[:, 3:5]].T)][:, :4]
        error_index = np.nonzero(np.sum(np.isnan(x_candidates), axis=1))
        if error_index[0].shape[0] > 0:
            print("--error!!!")
            q_candidates = np.delete(q_candidates, error_index, axis=0)
            q_ae = np.delete(q_ae, error_index, axis=0)
            phi_candidates = np.delete(phi_candidates, error_index, axis=0)
            x_candidates = np.delete(x_candidates, error_index, axis=0)

        # calculate alpha
        # (beta, dis)
        beta = np.arctan2(AB[1], AB[0])
        dis = np.linalg.norm(q_ae[:, :2], axis=1)
        alpha = (-np.arccos(np.clip((dis - x_candidates[:, 0] * np.cos(phi_candidates)) / b,
                                    -1, 1)) *
                 np.sign(phi_candidates) + beta)
        AE_alpha = np.arctan2(q_ae[:, 1], q_ae[:, 0])

        # use joint 0 to control alpha
        q_candidates[:, 0] += alpha - AE_alpha
        q_candidates[q_candidates[:, 0] > np.pi, 0] -= 2 * np.pi
        q_candidates[q_candidates[:, 0] < -np.pi, 0] += 2 * np.pi

        return q_candidates, phi_candidates, x_candidates

    def filter_candidates(self,
                          q_candidates,
                          phi_candidates,
                          x_candidates,
                          thres_r_ratio=None,
                          thres_phi_ratio=None,
                          thres_height=None):
        """
        Apply filters to the matched candidates to select the final set.
        :param q_candidates: The candidate configurations for q.
        :param phi_candidates: The candidate values for phi.
        :param x_candidates: The candidate values for x.
        :return: The filtered candidates for q, phi, x.
        """
        if thres_r_ratio is not None:
            self.thres_r_ratio = thres_r_ratio
        if thres_phi_ratio is not None:
            self.thres_phi_ratio = thres_phi_ratio
        if thres_height is not None:
            self.thres_height = thres_height

        # 1. Close to target wrt r
        sorted_indices = np.argsort(abs(x_candidates[:, 0]))  # Sort by distance to target
        n_total = len(sorted_indices)
        n_keep = max(1, int(n_total * self.thres_r_ratio))  # Keep top n_keep candidates
        final_indices = sorted_indices[:n_keep]

        q_candidates = q_candidates[final_indices]
        phi_candidates = phi_candidates[final_indices]
        x_candidates = x_candidates[final_indices]

        # 2. Safe height filtering
        height_mask = x_candidates[:, 1] > self.thres_height  # Filter by height
        q_candidates = q_candidates[height_mask]
        phi_candidates = phi_candidates[height_mask]
        x_candidates = x_candidates[height_mask]

        # 3. Filter based on large phi (joint 1 provides not much torque)
        if len(phi_candidates) > 0:
            sorted_phi_indices = np.argsort(-phi_candidates)  # Sort phi in descending order
            n_keep_phi = max(1,
                             int(len(sorted_phi_indices) * self.thres_phi_ratio))  # Keep top n_keep_phi candidates
            final_phi_indices = sorted_phi_indices[:n_keep_phi]

            q_candidates = q_candidates[final_phi_indices]
            phi_candidates = phi_candidates[final_phi_indices]
            x_candidates = x_candidates[final_phi_indices]

        # Print the number of valid candidates
        # print("Number of valid candidates:", q_candidates.shape[0])

        return q_candidates, phi_candidates, x_candidates

    def get_full_throwing_config(self, q, phi, x, posture=None):
        """
        Return full throwing configurations
        :input param robot description, q, phi, x
        :return: (q, phi, x, q_dot, blockPosInGripper, eef_velo, AE, box_position)
        calculate from the throwing aspect
        """
        r = x[0]
        z = x[1]
        r_dot = x[2]
        z_dot = x[3]

        # kinemetic forward
        AE, J = self.robot.forward(q, posture=posture)

        # in ee_site space
        throwing_angle = np.arctan2(AE[1], AE[0]) + phi
        EB_dir = np.array([np.cos(throwing_angle), np.sin(throwing_angle)])

        # get current jacobian to calculate q_dot = J_pinv @ eef_velo
        J_xyz = J[:3, :]
        J_xyz_pinv = np.linalg.pinv(J_xyz)

        eef_velo = np.array([EB_dir[0] * r_dot, EB_dir[1] * r_dot, z_dot])
        q_dot = J_xyz_pinv @ eef_velo
        EB = np.array([r * EB_dir[0], r * EB_dir[1], z])
        box_position = AE - EB

        # box_position = AE + np.array([-r * EB_dir[0], -r * EB_dir[1], -z]) # 3 dim

        # control last one joint to make end effector towards box

        gripperPos = self.robot.data.site("ee_site").xpos.copy() - 0.5 # position based on kuka_base
        gripperRot = self.robot.data.site("ee_site").xmat.copy().reshape(3,3)

        eef_velo_dir_3d = eef_velo / np.linalg.norm(eef_velo)

        tmp = AE + eef_velo_dir_3d
        blockPosInGripper = gripperRot.T @ (tmp - gripperPos)
        velo_angle_in_eef = np.arctan2(blockPosInGripper[1], blockPosInGripper[0])

        if (velo_angle_in_eef < math.pi) and (velo_angle_in_eef > -math.pi):
            eef_angle_near = velo_angle_in_eef
        elif velo_angle_in_eef > math.pi:
            eef_angle_near = velo_angle_in_eef - math.pi
        else:
            eef_angle_near = velo_angle_in_eef + math.pi

        q[-1] = eef_angle_near

        return (q, phi, x, q_dot, blockPosInGripper, eef_velo, AE, box_position)

    def generate_throw_config(self,
                              q_candidates,
                              phi_candidates,
                              x_candidates,
                              base0,
                              qs = None,
                              qs_dot = None,
                              posture=None,
                              joint_velocity_limits=None,
                              simulation=True):
        """
        Trajectory filter function
        input: q_candidates, phi_candidates, x_candidates, base0(box_position in xoy)
        :param joint_velocity_limits:
        {
            'max_abs': [1.5, 0.6, None, None, None, 2.0, None],
            'min_abs': [0.5, None, None, None, None, 1.0, None]
        }
        output: trajs, throw_configs
        """

        if simulation:
            q_cur = self.q0
            q_cur_dot = self.q0_dot
        else:
            q_cur = qs
            q_cur_dot = qs_dot
        n_candidates = q_candidates.shape[0]

        # get full throwing configuration and trajectories
        traj_durations =[]
        trajs = []
        throw_configs = []

        if joint_velocity_limits is None:
            joint_velocity_limits = {
                'max_abs': self.max_velocity.tolist(),
                'min_abs': [None] * 7
            }

        joint_limit_counters = {
            'max': [0] * 7,
            'min': [0] * 7
        }

#######################################################################################################################################
################################################ filter trajectory ####################################################################
        # record bad trajectories
        num_outlimit, num_ruckiger, num_small_deviation, num_vel_limit = 0, 0, 0, 0

        for i in range(n_candidates):
            candidate_idx = i

            # 1.check joint0 limitation
            q0 = q_candidates[candidate_idx][0]
            if q0 > self.q_ul[0] or q0 < self.q_ll[0]:
                num_outlimit += 1
                continue

            throw_config_full = self.get_full_throwing_config(q_candidates[candidate_idx],
                                                              phi_candidates[candidate_idx],
                                                              x_candidates[candidate_idx],
                                                              posture=posture)

            # 2 filter velocity joint provider
            q_dot = throw_config_full[3]
            skip_candidate = False
            for joint_idx in range(len(q_dot)):
                vel = q_dot[joint_idx]
                max_limit = joint_velocity_limits['max_abs'][joint_idx]
                min_limit = joint_velocity_limits['min_abs'][joint_idx]

                if max_limit is not None and abs(vel) > max_limit:
                    joint_limit_counters['max'][joint_idx] += 1
                    skip_candidate = True

                if min_limit is not None and abs(vel) < min_limit:
                    joint_limit_counters['min'][joint_idx] += 1
                    skip_candidate = True

            if skip_candidate:
                num_vel_limit += 1
                continue

            # 3. valid trajectory
            try:
                traj_throw = self.get_traj_from_ruckig(q0=q_cur, q0_dot=q_cur_dot,
                                                       qd=throw_config_full[0],
                                                       qd_dot=throw_config_full[3])

                if traj_throw.duration < 1e-10:
                    num_ruckiger += 1
                    continue

            except Exception as e:
                num_ruckiger += 1
                continue

            deviation = throw_config_full[-1][:2] - base0

            if np.linalg.norm(deviation) < 0.1:
                num_small_deviation += 1

            traj_durations.append(traj_throw.duration)
            trajs.append(traj_throw)
            throw_configs.append(throw_config_full)

        if len(trajs) == 0:
            print(f"Filter Summary: Total candidates: {n_candidates}, Out of limit: {num_outlimit}, Velocity limit: {num_vel_limit}, Ruckig error: {num_ruckiger}")
            for j in range(7):
                if joint_limit_counters['max'][j] > 0 or joint_limit_counters['min'][j] > 0:
                    print(f"  Joint {j}: Exceed Max: {joint_limit_counters['max'][j]}, Below Min: {joint_limit_counters['min'][j]}")

        print(f"    - generate_throw_config summary: input {n_candidates}, output {len(trajs)} (OutLimit: {num_outlimit}, VelLimit: {num_vel_limit}, RuckigErr: {num_ruckiger})")
        return trajs, throw_configs


    def get_traj_from_ruckig(self, q0, q0_dot,
                             qd, qd_dot,
                             margin_velocity=1.0, margin_acceleration=0.7,
                             margin_jerk=None):

        """
            Generates a smooth trajectory using the Ruckig algorithm for the given joint states.

            Parameters:
            - q0: Initial joint positions
            - q0_dot: Initial joint velocities
            - q0_dotdot: Initial joint accelerations
            - qd: Target joint positions
            - qd_dot: Target joint velocities
            - qd_dotdot: Target joint accelerations
            - margin_velocity: Velocity margin
            - margin_acceleration: Acceleration margin
            - margin_jerk: Jerk margin

            Returns:
            - trajectory: The generated trajectory object
            """

        if margin_jerk is None:
            margin_jerk = self.MARGIN_JERK


        input_length = len(q0)
        inp = InputParameter(input_length)
        inp.current_position = q0
        inp.current_velocity = q0_dot
        inp.current_acceleration = np.zeros(input_length)

        inp.target_position = qd
        inp.target_velocity = qd_dot
        inp.target_acceleration = np.zeros(input_length)

        inp.max_velocity = np.array(self.max_velocity * margin_velocity)
        inp.max_acceleration = np.array(self.max_acceleration * margin_acceleration)
        inp.max_jerk = np.array(self.max_jerk * margin_jerk)

        otg = Ruckig(input_length)
        trajectory = Trajectory(input_length)
        _ = otg.calculate(inp, trajectory)

        return trajectory


    def solve(self, animate=False,
              posture=None,
              box_pos=None,
              q0=None,
              q0_dot=None):
        q0 = q0 if q0 is not None else self.q0
        q0_dot = q0_dot if q0_dot is not None else self.q0_dot
        base0 = self.box_position[:2] if box_pos is None else box_pos[:2]
        # search result for specific posture
        q_candidates, phi_candidates, x_candidates = self.brt_robot_data_matching(posture, box_pos=box_pos)
        q_candidates, phi_candidates, x_candidates = self.filter_candidates(q_candidates,
                                                                            phi_candidates,
                                                                            x_candidates,
                                                                            thres_r_ratio=self.thres_r_ratio/2)
        if len(q_candidates) == 0:
            print("No result found")
            return 0

        trajs, throw_configs = self.generate_throw_config(
                                q_candidates,
                                phi_candidates,
                                x_candidates,
                                base0,
                                qs=q0,
                                qs_dot=q0_dot,
                                posture=posture)

        if len(trajs) == 0:
            print("No trajectory found")
            return 0


        # select the minimum-time trajectory
        traj_durations = [traj.duration for traj in trajs]
        selected_idx = np.argmin(traj_durations)
        traj_throw = trajs[selected_idx]
        throw_config_full = throw_configs[selected_idx]

        # print("box_position: ", self.box_position)
        # print("AB          : ", throw_config_full[-1])
        # print("deviation   : ", throw_config_full[-1] - self.box_position)
        # print("throwing state: ", throw_config_full[2])

        ref_trej = self.process_trajectory(traj_throw)

        if animate:
            self.throw_simulation_mujoco(ref_trej, throw_config_full, posture=posture)
        else:
            return traj_throw, throw_config_full

    def multi_waypoint_solve(self, box_positions, animate=True, full_search=False, q0=None):
        q0 = q0 if q0 is not None else self.q0
        start = time.time()
        base1 = box_positions[0][:2]
        base2 = box_positions[1][:2]

        q_candidates_1, phi_candidates_1, x_candidates_1 = (
            self.brt_robot_data_matching(posture='posture1', box_pos=box_positions[0]))
        if full_search:
            q_candidates_1, phi_candidates_1, x_candidates_1 = self.filter_candidates(q_candidates_1,
                                                                                      phi_candidates_1,
                                                                                      x_candidates_1,
                                                                                      thres_r_ratio=1.0,
                                                                                      thres_phi_ratio=1.0,
                                                                                      thres_height=-0.5)
        else:
            q_candidates_1, phi_candidates_1, x_candidates_1 = self.filter_candidates(q_candidates_1,
                                                                                      phi_candidates_1,
                                                                                      x_candidates_1,
                                                                                      thres_phi_ratio=self.thres_phi_ratio / 3
                                                                                      )
        q_candidates_2, phi_candidates_2, x_candidates_2 = (
            self.brt_robot_data_matching(posture='posture2', box_pos=box_positions[1]))
        if full_search:
            q_candidates_2, phi_candidates_2, x_candidates_2 = self.filter_candidates(q_candidates_2,
                                                                                      phi_candidates_2,
                                                                                      x_candidates_2,
                                                                                      thres_r_ratio=1.0,
                                                                                      thres_phi_ratio=1.0,
                                                                                      thres_height=-0.5,
                                                                                      )
        else:
            q_candidates_2, phi_candidates_2, x_candidates_2 = self.filter_candidates(q_candidates_2,
                                                                                      phi_candidates_2,
                                                                                      x_candidates_2)
        if len(q_candidates_1) == 0 or len(q_candidates_2) == 0:
            # print("No result found")
            return None, None, None

        if full_search:
            # joint_velocity_limits = {
            #     'max_abs': [1.5, 0.6, None, None, None, 2.0, None],
            #     'min_abs': [None, 0.0, None, None, None, 1.0, None]
            # }
            joint_velocity_limits = {
                'max_abs': [None, None, None, None, None, None, None],
                'min_abs': [None, None, None, None, None, None, None]
            }

            trajs_1, throw_configs_1 = self.generate_throw_config(q_candidates_1,
                                                                  phi_candidates_1,
                                                                  x_candidates_1,
                                                                  base1,
                                                                  qs=q0,
                                                                  posture='posture1',
                                                                  joint_velocity_limits=joint_velocity_limits)

            trajs_2, throw_configs_2 = self.generate_throw_config(q_candidates_2,
                                                                  phi_candidates_2,
                                                                  x_candidates_2,
                                                                  base2,
                                                                  posture='posture2',
                                                                  joint_velocity_limits=joint_velocity_limits)
        else:
            trajs_1, throw_configs_1 = self.generate_throw_config(q_candidates_1,
                                                                  phi_candidates_1,
                                                                  x_candidates_1,
                                                                  base1,
                                                                  qs=q0,
                                                                  posture='posture1')

            trajs_2, throw_configs_2 = self.generate_throw_config(q_candidates_2,
                                                                  phi_candidates_2,
                                                                  x_candidates_2,
                                                                  base2,
                                                                  posture='posture2')

        if len(trajs_1) == 0 or len(trajs_2) == 0:
            # print("No trajectory found")
            return None, None, None

        best_pair = None
        all_pairs = []
        all_pair_configs = []

        del q_candidates_1, phi_candidates_1, x_candidates_1
        del q_candidates_2, phi_candidates_2, x_candidates_2
        del trajs_1, trajs_2
        gc.collect()

        for i, cfg1 in enumerate(throw_configs_1):
            for j, cfg2 in enumerate(throw_configs_2):
                all_pairs.append(np.array([[cfg1[0], cfg1[3]], [cfg2[0], cfg2[3]]]))
                all_pair_configs.append((cfg1, cfg2))


        # 2. choose the shortest trajectory
        intermediate_time_all, final_trajectory_all = self.concatenate_all_pairs(all_pairs, q0=q0)
        min_duration_group = min(
            zip(intermediate_time_all, final_trajectory_all, all_pair_configs),
            key=lambda x: len(x[1]['timestamp'])
        )
        intermediate_time, final_trajectory, best_throw_config_pair = min_duration_group
        exe_time = final_trajectory["timestamp"][-1]
        time_1 = time.time()
        computation_time = time_1 - start
        print(f"Greedy search: computation time:{computation_time}, exe time:{exe_time}")

        if animate:
            self.throw_simulation_mujoco(final_trajectory, best_throw_config_pair,
                                     intermediate_time=intermediate_time)
        else:
            return final_trajectory, best_throw_config_pair, intermediate_time

    def solve_multi_targets(self, box_positions, postures=None, animate=True, full_search=False, q0=None, random_select=False, k=30):
        q0 = q0 if q0 is not None else self.q0
        n_targets = len(box_positions)
        pos_sort = k
        if postures is None:
            postures = ['posture1'] * n_targets
            
        start_time = time.time()
        
        target_pools = []
        for i in range(n_targets):
            pos = box_positions[i]
            posture = postures[i]
            base = pos[:2]
            
            q_c, phi_c, x_c = self.brt_robot_data_matching(posture=posture, box_pos=pos)
            q_c, phi_c, x_c = self.filter_candidates(q_c, phi_c, x_c, thres_r_ratio=1.0, thres_phi_ratio=1.0)
            
            if len(q_c) == 0:
                print(f"target {i} candidates is empty")
                return None, None, None
                
            _, throw_configs = self.generate_throw_config(
                q_c, phi_c, x_c, base,
                qs=q0, qs_dot=np.zeros(7), 
                posture=posture, simulation=False
            )
            
            if len(throw_configs) == 0:
                print(f"target {i} config pool is empty")
                return None, None, None
            
            target_pools.append(throw_configs)

        active_paths = []
        pool0 = target_pools[0]
        
        # Calculate distances from initial state q0
        q_cand = np.array([cfg[0] for cfg in pool0])
        qdot_cand = np.array([cfg[3] for cfg in pool0])
        distances = (np.linalg.norm(q_cand - q0, axis=1, ord=2) + 
                     2.0 * np.linalg.norm(qdot_cand - np.zeros(7), axis=1))
        
        sorted_idx = np.argsort(distances)
        candidates0 = [pool0[idx] for idx in sorted_idx[:pos_sort]]
        
        for cfg in candidates0:
            try:
                t = self.get_traj_from_ruckig(q0, np.zeros(7), cfg[0], cfg[3])
                if t and t.duration > 1e-10:
                    active_paths.append(([cfg], [t], t.duration))
            except:
                continue
        
        print(f"Layer 0: Kept {len(active_paths)} top-{pos_sort} valid configurations for Target 0")

        for i in range(1, n_targets):
            new_paths = []
            pool = target_pools[i]
            
            for path_configs, path_trajs, total_dur in active_paths:
                q_ref = path_configs[-1][0]
                qdot_ref = path_configs[-1][3]
                
                q_cand = np.array([cfg[0] for cfg in pool])
                qdot_cand = np.array([cfg[3] for cfg in pool])
                distances = (np.linalg.norm(q_cand - q_ref, axis=1, ord=2) + 
                             2.0 * np.linalg.norm(qdot_cand - qdot_ref, axis=1))
                
                sorted_idx = np.argsort(distances)
                candidates = [pool[idx] for idx in sorted_idx[:pos_sort]]
                
                for cand in candidates:
                    try:
                        t = self.get_traj_from_ruckig(q_ref, qdot_ref, cand[0], cand[3])
                        if t and t.duration > 1e-10:
                            new_paths.append((path_configs + [cand], path_trajs + [t], total_dur + t.duration))
                    except:
                        continue
            
            active_paths = new_paths
            if not active_paths:
                print(f"Target {i}: No valid path found from previous steps")
                return None, None, None
            
            print(f"Layer {i}: Generated {len(active_paths)} candidate paths")

        best_path = min(active_paths, key=lambda x: x[2])
        best_configs, best_trajs, best_duration = best_path
        
        total_traj = {
            'timestamp': np.array([]),
            'position': np.empty((0, 7)),
            'velocity': np.empty((0, 7)),
            'acceleration': np.empty((0, 7))
        }
        current_offset = 0.0
        intermediate_times = []
        for t_obj in best_trajs:
            seg = self.process_trajectory(t_obj, current_offset)
            if len(total_traj['timestamp']) > 0:
                start_idx = 1 if np.isclose(seg['timestamp'][0], total_traj['timestamp'][-1]) else 0
            else:
                start_idx = 0
            total_traj['timestamp'] = np.concatenate([total_traj['timestamp'], seg['timestamp'][start_idx:]])
            total_traj['position'] = np.concatenate([total_traj['position'], seg['position'][start_idx:]], axis=0)
            total_traj['velocity'] = np.concatenate([total_traj['velocity'], seg['velocity'][start_idx:]], axis=0)
            total_traj['acceleration'] = np.concatenate([total_traj['acceleration'], seg['acceleration'][start_idx:]], axis=0)
            current_offset += t_obj.duration
            intermediate_times.append(current_offset)

        search_time = time.time()-start_time
        print(f"Tree Search complete. Best duration: {best_duration:.2f}s, search time: {search_time:.2f}s, paths explored: {len(active_paths)}")
        
        total_energy = self.calculate_energy(total_traj)
        print(f"Total energy consumption: {total_energy:.2f} J")

        if animate:
            self.throw_simulation_mujoco_generic(total_traj, best_configs, intermediate_times, postures)
        
        return total_traj, best_configs, intermediate_times, search_time, best_duration, total_energy


    def throw_simulation_mujoco_generic(self,
                                        ref_sequence,
                                          throw_configs,
                                            intermediate_times,
                                              postures,
                                                speed_up=4):
        ROBOT_BASE_HEIGHT = 0.5
        n_targets = len(throw_configs)
        
        for i in range(n_targets):
            try:
                box_name = f"box{i+1}"
                sphere_name = f"sphere{i+1}"
                
                target_id = self.robot.model.body(box_name).id
                box_pos = throw_configs[i][-1].copy()
                box_pos[2] += ROBOT_BASE_HEIGHT
                self.robot._set_object_position(target_id, box_pos)
            except Exception:
                pass

        delta_t = 1.0 / self.freq
        physics_steps = int(speed_up * delta_t / self.robot.model.opt.timestep)
        
        q0 = ref_sequence['position'][0].copy()
        self.robot._set_joints(q0.tolist())
        self.robot.step_physics(steps=1, render=True)
        
        plan_time = ref_sequence['timestamp'][-1]
        sim_time = plan_time + 1.0 
        max_step = int(sim_time * self.freq)
        final_step = int(plan_time * self.freq)
        
        target_steps = [int(t * self.freq) for t in intermediate_times]
        milestones = sorted(list(set(target_steps + [final_step])))
        
        cur_step = 0
        prev_sim_step = 0
        while cur_step < max_step:
            execute_step = cur_step
            for m in milestones:
                if cur_step < m < cur_step + speed_up:
                    execute_step = m
                    break
            
            step_idx = min(execute_step, final_step - 1)
            if execute_step < final_step:
                pos = ref_sequence['position'][step_idx]
                vel = ref_sequence['velocity'][step_idx]
            else:
                pos = ref_sequence['position'][-1]
                vel = ref_sequence['velocity'][-1]
            
            self.robot._set_joints(pos, vel)
            
            current_target_idx = n_targets
            for i, step in enumerate(target_steps):
                if execute_step < step:
                    current_target_idx = i
                    break
            
            self.robot._set_hand_joints(self.robot.envelop_pose.copy().tolist())
            
            for i in range(n_targets):
                if i >= current_target_idx: 
                    try:
                        sphere_name = f"sphere{i+1}"
                        sphere_id = self.robot.model.body(sphere_name).id
                        posture = postures[i]
                        
                        if posture == 'posture1':
                            ee_pos = (self.robot.obj_x2base("thumb_site") + self.robot.obj_x2base("middle_site")) / 2
                            ee_vel = (self.robot.obj_v("thumb_site") + self.robot.obj_v("middle_site")) / 2
                        else:
                            ee_pos = (self.robot.obj_x2base("index_site") + self.robot.obj_x2base("ring_site")) / 2
                            ee_vel = (self.robot.obj_v("index_site") + self.robot.obj_v("ring_site")) / 2
                            
                        ee_pos[2] += ROBOT_BASE_HEIGHT
                        self.robot._set_object_position(sphere_id, ee_pos, ee_vel[:3])
                    except Exception:
                        pass

            step_diff = execute_step - prev_sim_step
            if step_diff > 0:
                p_steps = int(step_diff * (1.0 / self.freq) / self.robot.model.opt.timestep)
                if p_steps > 0:
                    self.robot.step_physics(steps=p_steps, render=True)
            else:
                self.robot.step_physics(steps=1, render=True)
            
            prev_sim_step = execute_step
            if execute_step == cur_step:
                cur_step += speed_up
            else:
                cur_step = execute_step
                
            time.sleep(delta_t)


    def calculate_energy(self, ref_sequence):
        """
        Calculate total energy consumption of a trajectory using Pinocchio's RNEA.
        Energy = integral(max(tau * q_dot, 0)) dt
        """
        if ref_sequence is None or len(ref_sequence['timestamp']) == 0:
            return 0.0
            
        energy = 0.0
        dt = 1.0 / self.freq
        
        for i in range(len(ref_sequence['timestamp'])):
            q = ref_sequence['position'][i]
            v = ref_sequence['velocity'][i]
            a = ref_sequence['acceleration'][i]
            
            # RNEA to get joint torques
            tau = pin.rnea(self.pin_model, self.pin_data, q, v, a)
            
            # Mechanical power P = sum(tau_i * v_i)
            # We only consider positive work (power consumed)
            power = np.dot(tau, v)
            energy += max(power, 0) * dt
            
        return energy

    def process_trajectory(self, traj, time_offset=0.0):

        num_points = int(round(traj.duration * self.freq)) + 1
        time_points = np.linspace(0, traj.duration, num_points)

        n_dof = traj.degrees_of_freedom
        segment = {
            'timestamp': time_points + time_offset,
            'position': np.empty((num_points, n_dof)),
            'velocity': np.empty((num_points, n_dof)),
            'acceleration': np.empty((num_points, n_dof))
        }

        for i, t in enumerate(time_points):
            segment['position'][i], segment['velocity'][i], segment['acceleration'][i] = traj.at_time(t)

        return segment

    def concatenate_trajectories(self, traj1, traj2):

        ref_sequence = {
            'timestamp': np.array([]),
            'position': np.empty((0, traj1.degrees_of_freedom)),
            'velocity': np.empty((0, traj1.degrees_of_freedom)),
            'acceleration': np.empty((0, traj1.degrees_of_freedom))
        }

        segment1 = self.process_trajectory(traj1)
        if segment1 is not None:
            ref_sequence['timestamp'] = segment1['timestamp']
            ref_sequence['position'] = segment1['position']
            ref_sequence['velocity'] = segment1['velocity']
            ref_sequence['acceleration'] = segment1['acceleration']

        time_offset = traj1.duration if traj1 else 0.0
        segment2 = self.process_trajectory(traj2, time_offset)

        if segment2 is not None:
            if len(ref_sequence['timestamp']) > 0:
                overlap = np.isclose(segment2['timestamp'][0], ref_sequence['timestamp'][-1])
                start_idx = 1 if overlap else 0
            else:
                start_idx = 0

            ref_sequence['timestamp'] = np.concatenate([
                ref_sequence['timestamp'],
                segment2['timestamp'][start_idx:]
            ])
            ref_sequence['position'] = np.concatenate([
                ref_sequence['position'],
                segment2['position'][start_idx:]
            ], axis=0)
            ref_sequence['velocity'] = np.concatenate([
                ref_sequence['velocity'],
                segment2['velocity'][start_idx:]
            ], axis=0)
            ref_sequence['acceleration'] = np.concatenate([
                ref_sequence['acceleration'],
                segment2['acceleration'][start_idx:]
            ], axis=0)

        return time_offset, ref_sequence

    def concatenate_all_pairs(self, all_pairs, q0=None):
        q0 = q0 if q0 is not None else self.q0
        intermediate_time_all = []
        final_trajectory_all = []
        for i, pair in enumerate(all_pairs):
            traj_1 = self.get_traj_from_ruckig(q0, np.zeros(7),
                                                pair[0][0], pair[0][1])
            traj_2 = self.get_traj_from_ruckig(pair[0][0], pair[0][1], pair[1][0], pair[1][1])
            intermediate_time, final_trajectory = self.concatenate_trajectories(traj_1, traj_2)
            intermediate_time_all.append(intermediate_time)
            final_trajectory_all.append(final_trajectory)

        return intermediate_time_all, final_trajectory_all

    def throw_simulation_mujoco(self, ref_sequence,
                                throw_config_full=None,
                                intermediate_time=None,
                                posture =None,
                                speed_up=50):
        ROBOT_BASE_HEIGHT = 0.5
        if throw_config_full is not None:
            # ... (keep existing logic)
            if len(throw_config_full) == 2:
                box_position_1 = throw_config_full[0][-1].copy()
                box_position_2 = throw_config_full[1][-1].copy()

                # set the target box position for visualization
                target_id_1 = self.robot.model.body("box1").id
                target_id_2 = self.robot.model.body("box2").id
                target_position_1 = box_position_1
                target_position_1[2] += ROBOT_BASE_HEIGHT
                target_position_2 = box_position_2
                target_position_2[2] += ROBOT_BASE_HEIGHT
                self.robot._set_object_position(target_id_1, target_position_1)
                self.robot._set_object_position(target_id_2, target_position_2)
            else:
                # set the target box position for visualization
                target_id = self.robot.model.body("box1").id
                target_position = self.box_position.copy()
                target_position[2] += ROBOT_BASE_HEIGHT
                self.robot._set_object_position(target_id, target_position)


        delta_t = 1.0 / self.freq
        physics_steps = int(speed_up * delta_t / self.robot.model.opt.timestep)

        # reset the joint
        q0 = ref_sequence['position'][0].copy()
        self.robot._set_joints(q0.tolist())
        self.robot.step_physics(steps=1, render=True)

        plan_time = ref_sequence['timestamp'][-1]
        sim_time = 12.0
        max_step =  max(int(plan_time * self.freq) + 1, int(sim_time * self.freq))
        intermediate_step = int(intermediate_time * self.freq) if intermediate_time is not None else None
        final_step = int(plan_time * self.freq)

        control_mode = 'single_object'
        if intermediate_time is not None:
            control_mode = 'multi_object'

        cur_step = 0
        flag = True
        ee_pos = {
            'posture1': np.array([]),
            'posture2': np.array([])
        }

        ee_vel = {
            'posture1': np.array([]),
            'posture2': np.array([])
        }
        while True:
            step_idx = min(cur_step, final_step - 1)
            if flag:
                pos = ref_sequence['position'][step_idx].copy()
                vel = ref_sequence['velocity'][step_idx].copy()
                self.robot._set_joints(pos, vel)
            else:
                pos = ref_sequence['position'][-1].copy()
                vel = ref_sequence['velocity'][-1].copy()
                self.robot._set_joints(pos, vel)

            # get the state of ee_site in the frame of kuka_base
            object1_id = self.robot.model.body("sphere1").id
            object2_id = self.robot.model.body("sphere2").id

            ee_pos['posture1'] = (self.robot.obj_x2base("thumb_site") + self.robot.obj_x2base("middle_site")) / 2
            ee_pos['posture1'][2] += ROBOT_BASE_HEIGHT
            ee_vel['posture1'] = (self.robot.obj_v("thumb_site")+self.robot.obj_v("middle_site")) / 2

            ee_pos['posture2'] = (self.robot.obj_x2base("index_site") + self.robot.obj_x2base("ring_site")) / 2
            ee_pos['posture2'][2] += ROBOT_BASE_HEIGHT
            ee_vel['posture2'] = (self.robot.obj_v("index_site") + self.robot.obj_v("ring_site")) / 2

            if control_mode == 'multi_object':
                if cur_step < intermediate_step:
                    self.robot._set_hand_joints(self.robot.envelop_pose.copy().tolist())
                    self.robot._set_object_position(object1_id, ee_pos['posture1'], ee_vel['posture1'][:3])
                    self.robot._set_object_position(object2_id, ee_pos['posture2'], ee_vel['posture2'][:3])
                elif cur_step >= intermediate_step and cur_step < final_step:
                    self.robot._set_hand_joints(self.robot.envelop_pose.copy().tolist())
                    self.robot._set_object_position(object2_id, ee_pos['posture2'], ee_vel['posture2'][:3])
            else:
                if cur_step < final_step:
                    self.robot._set_hand_joints(self.robot.envelop_pose.copy().tolist())
                    self.robot._set_object_position(object1_id, ee_pos[posture], ee_vel[posture][:3])
                else:
                    self.robot._set_hand_joints(self.robot.hand_home_pose.copy().tolist())

            self.robot.step_physics(steps=physics_steps, render=True)
            cur_step += speed_up
            if cur_step > final_step:
                flag = False
            time.sleep(delta_t)

            if cur_step > max_step:
                break

    def KD_Tree_search(self, target_boxes):
        target_boxes = target_boxes.copy().flatten()
        X_path = self.off_data_path + '/training_X.npy'
        Y_path = self.off_data_path + '/training_Y.npy'
        Keys = np.load(X_path)
        Values = np.load(Y_path)

        tree = KDTree(Keys)

        start = time.time()

        dist, idx = tree.query(target_boxes)

        matched_value = Values[idx]

        qA = matched_value[:7]
        qA_dot = matched_value[7:14]
        qB = matched_value[14:21]
        qB_dot = matched_value[21:]

        time_1 = time.time()


        traj_1 = self.get_traj_from_ruckig(self.q0, self.q0_dot,
                                           qA, qA_dot)
        traj_2 = self.get_traj_from_ruckig(qA, qA_dot, qB, qB_dot)
        intermediate_time, final_trajectory = self.concatenate_trajectories(traj_1, traj_2)
        config_full = ([target_boxes[:3]], [target_boxes[3:]])
        time_2 = time.time()

        computation_time = time_2 - start
        exe_time = final_trajectory["timestamp"][-1]
        print(f"KD-Tree map computation time:{computation_time}, exe time:{exe_time}")
        del tree, Keys, Values
        gc.collect()

        self.throw_simulation_mujoco(final_trajectory, config_full,
                                         intermediate_time=intermediate_time)

    def naive_search(self, target_boxes, q0=None, simulation=True):
        """
        Sequential search for multiple targets: solve each target one by one, 
        starting from the end configuration of the previous one.
        """
        start = time.time()
        q0 = q0 if q0 is not None else self.q0
        current_q = q0
        current_q_dot = self.q0_dot
        
        all_trajs = []
        all_configs = []
        intermediate_times = []
        current_total_time = 0.0
        
        n_targets = len(target_boxes)
        
        for i in range(n_targets):
            box_pos = target_boxes[i]
            # Sequential solving, always using posture1 as requested
            res = self.solve(animate=False, posture="posture1", box_pos=box_pos, q0=current_q, q0_dot=current_q_dot)
            if res == 0 or res is None:
                print(f"NAIVE: No result found for target {i} at {box_pos}")
                return None, None, None
            
            traj, config = res
            all_trajs.append(traj)
            all_configs.append(config)
            
            # Update current state to the end of the solved trajectory
            current_q = config[0]
            current_q_dot = config[3]
            
            current_total_time += traj.duration
            intermediate_times.append(current_total_time)

        # Concatenate trajectories into a single dictionary format
        total_traj = {
            'timestamp': np.array([]),
            'position': np.empty((0, 7)),
            'velocity': np.empty((0, 7)),
            'acceleration': np.empty((0, 7))
        }
        
        offset = 0.0
        for t_obj in all_trajs:
            seg = self.process_trajectory(t_obj, offset)
            if len(total_traj['timestamp']) > 0:
                # Avoid duplicating the waypoint point
                start_idx = 1 if np.isclose(seg['timestamp'][0], total_traj['timestamp'][-1]) else 0
            else:
                start_idx = 0
            
            total_traj['timestamp'] = np.concatenate([total_traj['timestamp'], seg['timestamp'][start_idx:]])
            total_traj['position'] = np.concatenate([total_traj['position'], seg['position'][start_idx:]], axis=0)
            total_traj['velocity'] = np.concatenate([total_traj['velocity'], seg['velocity'][start_idx:]], axis=0)
            total_traj['acceleration'] = np.concatenate([total_traj['acceleration'], seg['acceleration'][start_idx:]], axis=0)
            
            offset += t_obj.duration

        computation_time = time.time() - start
        print(f"NAIVE: computation time:{computation_time:.4f}s, total exe time:{offset:.4f}s")

        if simulation:
            postures = ["posture1"] * n_targets
            self.throw_simulation_mujoco_generic(total_traj, all_configs, intermediate_times, postures)

        return total_traj, all_configs, intermediate_times


if __name__ == "__main__":
    q_min = np.array([-2.96705972839, -2.09439510239, -2.96705972839, -2.09439510239, -2.96705972839,
                      -2.09439510239, -3.05432619099])
    q_max = np.array([2.96705972839, 2.09439510239, 2.96705972839, 2.09439510239, 2.96705972839,
                      2.09439510239, 3.05432619099])
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    hedgehog_path = os.path.abspath(os.path.join(current_dir, '../hedgehog_data'))
    brt_path = os.path.abspath(os.path.join(current_dir, '../brt_data'))
    robot_path = os.path.abspath(os.path.join(current_dir, '../description/iiwa7_allegro_throwing.xml'))
    
    box_position = np.array([1.3, 0.07, -0.158])
    box1 = np.array([1.25, 0.35, -0.1])
    box2 = np.array([0.4, 1.3, -0.1])
    box_positions = np.array([box1, box2])

    trajectory_generator = TrajectoryGenerator(q_max, q_min,
                                               hedgehog_path, brt_path,
                                               robot_path,box_position)

    # trajectory_generator.solve(animate=True, posture="posture1")

    # naive search
    # trajectory_generator.naive_search(box_positions)

    # greedy search
    trajectory_generator.multi_waypoint_solve(box_positions, full_search=True)
    
    # KD-Tree search
    # trajectory_generator.KD_Tree_search(box_positions)