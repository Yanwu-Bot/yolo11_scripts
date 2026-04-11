#用于DTW的多维特征
from utill import *
import numpy as np

class Feature:
    def __init__(self, p_pos):
        '''
        p_pos:关键点坐标列表
        '''
        self.neck = [(p_pos[5][0] + p_pos[6][0])/2, (p_pos[5][1] + p_pos[6][1])/2]                 #脖颈
        self.hip_center = [(p_pos[11][0] + p_pos[12][0])/2, (p_pos[11][1] + p_pos[12][1])/2]       #髋中心
        self.thorax = [(self.neck[0] + self.hip_center[0])/2, (self.neck[1] + self.hip_center[1])/2]  #躯干中心
        self.nose = p_pos[0]                              #鼻子
        self.l_eye = p_pos[1]                             #左眼
        self.r_eye = p_pos[2]                             #右眼
        self.l_ear = p_pos[3]                             #左耳
        self.r_ear = p_pos[4]                             #右耳
        self.l_shoulder = p_pos[5]                        #左肩
        self.r_shoulder = p_pos[6]                        #右肩
        self.l_elbow = p_pos[7]                           #左肘
        self.r_elbow = p_pos[8]                           #右肘
        self.l_hand = p_pos[9]                            #左手
        self.r_hand = p_pos[10]                           #右手
        self.l_hip = p_pos[11]                            #左髋
        self.r_hip = p_pos[12]                            #右髋
        self.l_knee = p_pos[13]                           #左膝
        self.r_knee = p_pos[14]                           #右膝    
        self.l_foot = p_pos[15]                           #左脚
        self.r_foot = p_pos[16]                           #右脚
        
        shoulder_vec = np.array(self.r_shoulder) - np.array(self.l_shoulder)
        self.shoulder_width = float(np.linalg.norm(shoulder_vec))
        self.spine_vec = np.array(self.hip_center) - np.array(self.neck)
        self.spine_width = float(np.linalg.norm(self.spine_vec))
        self.front_vec = np.array([0, 1])  # 假设Y轴向前

    #获取中心点
    def get_main_center(self, point_list):
        x_sum = 0
        y_sum = 0
        for pair in point_list:
            x_sum += pair[0]
            y_sum += pair[1]
        x_avg = x_sum/len(point_list)
        y_avg = y_sum/len(point_list)
        point = [x_avg, y_avg]
        return point
    
    def get_part_angle(self):
        """
        夹角特征
        """
        angle_list = [
            calculate_angle(self.thorax, self.neck, self.r_shoulder),
            calculate_angle(self.thorax, self.neck, self.l_shoulder),
            calculate_angle(self.thorax, self.hip_center, self.r_shoulder),
            calculate_angle(self.thorax, self.hip_center, self.l_shoulder),
            calculate_angle(self.r_shoulder, self.thorax, self.r_elbow),
            calculate_angle(self.l_shoulder, self.thorax, self.l_elbow),
            calculate_angle(self.r_elbow, self.r_shoulder, self.r_hand),
            calculate_angle(self.l_elbow, self.l_shoulder, self.l_hand),            
            calculate_angle(self.r_shoulder, self.neck, self.r_elbow),
            calculate_angle(self.l_shoulder, self.neck, self.l_elbow),
            calculate_angle(self.hip_center, self.thorax, self.r_hip),
            calculate_angle(self.hip_center, self.thorax, self.l_hip),
            calculate_angle(self.r_hip, self.hip_center, self.r_knee),
            calculate_angle(self.l_hip, self.hip_center, self.l_knee),
            calculate_angle(self.r_knee, self.r_hip, self.r_foot),
            calculate_angle(self.l_knee, self.l_hip, self.l_foot),
            calculate_angle(self.r_elbow, self.r_shoulder, self.hip_center),
            calculate_angle(self.l_elbow, self.l_shoulder, self.hip_center),
            calculate_angle(self.r_knee, self.r_hip, self.neck),
            calculate_angle(self.l_knee, self.l_hip, self.neck),
            calculate_angle(self.l_shoulder, self.l_elbow, self.l_hand),
            calculate_angle(self.r_shoulder, self.r_elbow, self.r_hand),
            calculate_angle(self.r_hip, self.r_knee, self.r_foot),
            calculate_angle(self.l_hip, self.l_knee, self.l_foot),
        ]
        
        MAX_ANGLE = 180.0
        normalized_angles = [min(a / MAX_ANGLE,1) for a in angle_list]
        
        return normalized_angles
                
    def get_center(self):
        """
        获取基于脊柱长度归一化的中心点坐标
        返回: 8个归一化后的数值 [x1,y1,x2,y2,x3,y3,x4,y4]
        范围：通常在[-3, 3]之间，使用tanh压缩到[-1, 1]
        """
        try:
            # 计算原始中心点
            center1 = self.get_main_center([self.r_elbow, self.r_hand, self.r_shoulder])
            center2 = self.get_main_center([self.l_elbow, self.l_hand, self.l_shoulder])
            center3 = self.get_main_center([self.l_hip, self.l_knee, self.l_foot])
            center4 = self.get_main_center([self.r_hip, self.r_knee, self.r_foot])
            
            # 获取脊柱长度和参考点
            spine_length = self.spine_width
            if spine_length == 0:
                return [0.0] * 8
            
            # 使用脖子作为参考点
            ref_x, ref_y = self.neck[0], self.neck[1]
            
            # 归一化所有中心点，并用tanh压缩到[-1, 1]
            normalized = []
            for center in [center1, center2, center3, center4]:
                rel_x = (center[0] - ref_x) / spine_length
                rel_y = (center[1] - ref_y) / spine_length
                # 使用tanh压缩，避免极端值影响
                # tanh(3) ≈ 0.995，所以[-3,3]范围内的值会被映射到[-0.995, 0.995]
                normalized.extend([np.tanh(rel_x), np.tanh(rel_y)])
            
            return normalized
            
        except Exception as e:
            print(f"get_normalized_center错误: {e}")
            return [0.0] * 8

    def get_beta_features(self):
        """
        2维身体朝向特征
        返回: [β₁, β₂] 列表，范围都在[0, 1]
        """
        betas = []
        
        # β₁: 身体前倾角度，除以90归一化到[0, 1]
        vertical_point = [self.neck[0], self.neck[1] + 100]
        beta1 = calculate_angle(vertical_point, self.neck, self.hip_center)
        betas.append(min(beta1 / 90.0, 1.0))  # 0-90度 → 0-1
        
        # β₂: 两脚间距离，用脊柱归一化，然后压缩到[0, 1]
        dx = self.l_foot[0] - self.r_foot[0]
        dy = self.l_foot[1] - self.r_foot[1]
        foot_dist = (dx*dx + dy*dy) ** 0.5
        beta2_raw = foot_dist / (self.spine_width + 1e-6)
        # 假设最大步宽不会超过脊柱长度的2倍，使用tanh压缩
        betas.append(np.tanh(beta2_raw / 2.0))  # 映射到[0, 1)
        
        return betas

    def get_gamma_features(self):
        """
        跑步专用的4维对侧协调特征
        返回: [γ₁, γ₂, γ₃, γ₄] 列表，范围在[-1, 1]
        """
        # 计算四肢相对于躯干的位移（像素单位）
        left_arm_phase = self.l_hand[0] - self.l_shoulder[0]
        right_arm_phase = self.r_hand[0] - self.r_shoulder[0]
        left_leg_phase = self.l_foot[0] - self.l_hip[0]
        right_leg_phase = self.r_foot[0] - self.r_hip[0]
        
        # 使用脊柱宽度作为参考长度来归一化（使特征与人体尺度无关）
        spine_len = self.spine_width + 1e-6
        
        left_arm_norm = left_arm_phase / spine_len
        right_arm_norm = right_arm_phase / spine_len
        left_leg_norm = left_leg_phase / spine_len
        right_leg_norm = right_leg_phase / spine_len
        
        # 用tanh压缩到[-1, 1]，假设相对位移通常在[-2, 2]范围内
        gamma = [
            np.tanh(left_arm_norm),
            np.tanh(right_arm_norm),
            np.tanh(left_leg_norm),
            np.tanh(right_leg_norm)
        ]
        
        return gamma
    
    def get_all_features(self):
        """
        获取所有特征：24维角度 + 8维中心 + 2维beta + 4维gamma = 38维
        所有特征都已经归一化到[-1, 1]或[0, 1]范围
        返回: 38维列表
        """
        part_angles = self.get_part_angle()   # 24维，范围[0, 1]
        center = self.get_center()            # 8维，范围[-1, 1]
        beta = self.get_beta_features()       # 2维，范围[0, 1]
        gamma = self.get_gamma_features()     # 4维，范围[-1, 1]
        
        all_features = part_angles + center + beta + gamma
        return all_features