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
        
        self.spine_width = float(np.linalg.norm(np.array(self.hip_center) - np.array(self.neck)))

    #获取中心点
    def get_main_center(self, point_list):
        x_sum = 0
        y_sum = 0
        for pair in point_list:
            x_sum += pair[0]
            y_sum += pair[1]
        return [x_sum/len(point_list), y_sum/len(point_list)]
    
    def get_part_angle(self):
        """
        夹角特征
        """
        angle_list = [
            #手部
            calculate_angle_180(self.neck, self.r_shoulder, self.r_elbow),
            calculate_angle_180(self.neck, self.l_shoulder, self.l_elbow),
            calculate_angle_180(self.neck, self.r_shoulder, self.r_hand),
            calculate_angle_180(self.neck, self.l_shoulder, self.l_hand),
            #腿部
            calculate_angle_180(self.hip_center, self.r_hip, self.r_knee),
            calculate_angle_180(self.hip_center, self.l_hip, self.l_knee),
            calculate_angle_180(self.hip_center, self.r_hip, self.r_foot),
            calculate_angle_180(self.hip_center, self.l_hip, self.l_foot),
            #关节角
            calculate_angle_180(self.l_shoulder, self.l_elbow, self.l_hand),
            calculate_angle_180(self.r_shoulder, self.r_elbow, self.r_hand),
            calculate_angle_180(self.r_hip, self.r_knee, self.r_foot),
            calculate_angle_180(self.l_hip, self.l_knee, self.l_foot),
            calculate_angle_180(self.r_elbow, self.r_shoulder, self.r_hand),
            calculate_angle_180(self.l_elbow, self.l_shoulder, self.l_hand),
            calculate_angle_180(self.r_knee, self.r_hip, self.r_foot),
            calculate_angle_180(self.l_knee, self.l_hip, self.l_foot),
        ]
        return [min(a / 180.0, 1.0) for a in angle_list]
                
    def get_center(self):
        try:
            centers = [
                self.get_main_center([self.r_elbow, self.r_hand, self.r_shoulder]),
                self.get_main_center([self.l_elbow, self.l_hand, self.l_shoulder]),
                self.get_main_center([self.l_hip, self.l_knee, self.l_foot]),
                self.get_main_center([self.r_hip, self.r_knee, self.r_foot]),
            ]
            return [min(calculate_angle_180(self.neck, self.hip_center, c) / 180.0, 1.0) for c in centers]
        except Exception as e:
            print(f"get_normalized_center错误: {e}")
            return [0.0] * 4

    def get_beta_features(self):
        """
        2维身体朝向特征
        返回: [β₁, β₂] 列表，范围都在[0, 1]
        """
        # β₁: 身体前倾角度，除以90并裁剪到[0, 1]
        vertical_point = [self.neck[0], self.neck[1] + 100]
        beta1 = calculate_angle(vertical_point, self.hip_center, self.neck)
        beta1 = min(max(beta1 / 90.0, 0.0), 1.0)
        
        # β₂: 两脚间距离，用脊柱归一化后tanh压缩到[0, 1)
        dx = self.l_foot[0] - self.r_foot[0]
        dy = self.l_foot[1] - self.r_foot[1]
        foot_dist = (dx*dx + dy*dy) ** 0.5
        beta2 = np.tanh(foot_dist / (self.spine_width + 1e-6) / 2.0)
        
        return [beta1, beta2]

    def get_gamma_features(self):
        """
        跑步专用的4维对侧协调特征
        返回: [γ₁, γ₂, γ₃, γ₄] 列表，范围在[-1, 1]
        """
        spine_len = self.spine_width + 1e-6
        gamma = [
            np.tanh((self.l_hand[0] - self.l_shoulder[0]) / spine_len),
            np.tanh((self.r_hand[0] - self.r_shoulder[0]) / spine_len),
            np.tanh((self.l_foot[0] - self.l_hip[0]) / spine_len),
            np.tanh((self.r_foot[0] - self.r_hip[0]) / spine_len),
        ]
        return [(g + 1.0) / 2.0 for g in gamma]
    
    def get_all_features(self):
        return (self.get_part_angle() + self.get_center() +
                self.get_beta_features() + self.get_gamma_features())