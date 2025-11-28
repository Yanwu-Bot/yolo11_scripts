import math

def calculate_angle(point_a, point_b, point_c):
    """
    计算三个点之间的角度（点B为顶点）
    :param point_a: 点A的坐标 (x, y)
    :param point_b: 点B的坐标 (x, y)
    :param point_c: 点C的坐标 (x, y)
    :return: 角度（度数）
    """
    # 创建向量BA和BC
    ba = (point_a[0] - point_b[0], point_a[1] - point_b[1])
    bc = (point_c[0] - point_b[0], point_c[1] - point_b[1])
    
    # 计算点积
    dot_product = ba[0] * bc[0] + ba[1] * bc[1]
    
    # 计算向量模长
    magnitude_ba = math.sqrt(ba[0]**2 + ba[1]**2)
    magnitude_bc = math.sqrt(bc[0]**2 + bc[1]**2)
    
    # 计算余弦值
    cos_angle = dot_product / (magnitude_ba * magnitude_bc)
    
    # 防止浮点数精度问题导致的值超出[-1, 1]范围
    cos_angle = max(min(cos_angle, 1), -1)
    
    # 计算角度（弧度）并转换为度数
    angle_rad = math.acos(cos_angle)
    angle_deg = math.degrees(angle_rad)
    
    return angle_deg

# 示例使用
# point_a = (0, 0)
# point_b = (1, 0)
# point_c = (0, 1)

# angle = calculate_angle(point_a, point_b, point_c)
# print(f"角度为: {angle:.2f}°")