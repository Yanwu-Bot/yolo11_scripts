#用于DTW的多维特征
from utill import *

class Feature:
    def __init__(self,p_pos):
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
        self.front_vec = np.array([0, 1])  # 假设Y轴向前

    #获取中心点
    def get_main_center(self,point_list):
        x_sum = 0
        y_sum = 0
        for pair in point_list:
            x_sum += pair[0]
            y_sum += pair[1]
        x_avg = x_sum/len(point_list)
        y_avg = y_sum/len(point_list)
        point = [x_avg,y_avg]
        return point
    
    
    def get_part_angle(self):
        """
        夹角
        """
        angle = [
            calculate_angle(self.thorax,self.neck,self.r_shoulder),
            calculate_angle(self.thorax,self.hip_center,self.r_shoulder),
            calculate_angle(self.r_shoulder,self.thorax,self.r_elbow),
            calculate_angle(self.r_elbow,self.r_shoulder,self.r_hand),
            calculate_angle(self.l_shoulder,self.neck,self.l_elbow),
            calculate_angle(self.l_elbow,self.l_shoulder,self.l_hand),
            calculate_angle(self.hip_center,self.thorax,self.r_hip),
            calculate_angle(self.r_hip,self.hip_center,self.r_knee),
            calculate_angle(self.r_knee,self.r_hip,self.r_foot),
            calculate_angle(self.l_hip,self.hip_center,self.l_knee),
            calculate_angle(self.l_knee,self.l_hip,self.l_foot),
            calculate_angle(self.r_elbow,self.r_shoulder,self.hip_center),
            calculate_angle(self.l_elbow,self.l_shoulder,self.hip_center),
            calculate_angle(self.r_knee,self.r_hip,self.neck),
            calculate_angle(self.l_knee,self.l_hip,self.neck),
            calculate_angle(self.l_shoulder,self.l_elbow,self.l_hand),
            calculate_angle(self.r_shoulder,self.r_elbow,self.r_hand)
        ]
        
        return angle
    
    def get_center(self):
        """
        上肢块中心，下肢块中心
        """
        center1 = [self.r_elbow,self.r_hand,self.r_shoulder]       #右臂
        center2 = [self.l_elbow,self.l_hand,self.l_shoulder]       #左臂
        center3 = [self.l_hip,self.l_knee,self.l_foot]             #左腿
        center4 = [self.r_hip,self.r_knee,self.r_foot]             #右腿
        center = [
            self.get_main_center(center1),
            self.get_main_center(center2),
            self.get_main_center(center3),
            self.get_main_center(center4)
        ]

        return center

    def get_beta_features(self):
        """
        4维身体朝向特征
        返回: [β₁, β₂, β₃, β₄] 列表
        """
        betas = []
        # β₁: 身体前倾角度 
        # 计算: 脊柱向量与垂直轴的夹角
        vertical_point = [self.neck[0], self.neck[1] + 100]   #脖子正下方的点
        beta1 = calculate_angle(vertical_point,self.neck,self.hip_center)
        betas.append(beta1)  # 0-90度，越小越直立
        
        # β₂: 两脚间距离
        dx = self.l_foot[0] - self.r_foot[0]  # x坐标差
        dy = self.l_foot[1] - self.r_foot[1]  # y坐标差
        foot_dist = (dx*dx + dy*dy) ** 0.5     # 欧氏距离 = √(dx² + dy²)
        beta2 = foot_dist / (self.shoulder_width + 1e-6)  # 用肩宽归一化
        betas.append(beta2)
        
        return betas

    def get_gamma_features(self, prev_frame=None):
        """
        跑步专用的4维对侧协调特征
        返回: [γ₁, γ₂, γ₃, γ₄] 列表
        γ₁: 左臂相位
        γ₂: 右臂相位  
        γ₃: 左腿相位
        γ₄: 右腿相位
        """
        # 计算四肢的"相位"（前/后位置）
        
        # 左臂相位: 左手相对于左肩的前后位置
        left_arm_phase = (self.l_hand[0] - self.l_shoulder[0])  # X轴前后
        
        # 右臂相位
        right_arm_phase = (self.r_hand[0] - self.r_shoulder[0])
        
        # 左腿相位: 左脚相对于左髋的前后位置
        left_leg_phase = (self.l_foot[0] - self.l_hip[0])
        
        # 右腿相位
        right_leg_phase = (self.r_foot[0] - self.r_hip[0])
        
        # 归一化到[-1, 1]范围
        max_val = max(abs(left_arm_phase), abs(right_arm_phase), 
                    abs(left_leg_phase), abs(right_leg_phase)) + 1e-6
        
        gamma = [
            left_arm_phase / max_val,
            right_arm_phase / max_val,
            left_leg_phase / max_val,
            right_leg_phase / max_val
        ]
        
        return gamma
    
    def get_all_features(self):
        """
        获取所有特征：原来的17维 + 中心 + β(2维) + γ(4维) = 27维
        返回: 列表
        """
        part_angles = self.get_part_angle()   # 17维列表
        center = self.get_center()            # 4维列表
        beta = self.get_beta_features()       # 2维列表
        gamma = self.get_gamma_features()     # 4维列表
        
        all_features = part_angles + center + beta + gamma  # 列表拼接
        return all_features  # 27维列表