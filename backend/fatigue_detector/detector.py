import random  # 实际项目中替换为真实算法

def detect_fatigue(image_path):
    """模拟算法检测逻辑，实际项目应调用真实算法"""
    # 示例返回结构
    return {
        'state': '疲劳' if random.random() > 0.5 else '清醒',
        'confidence': round(random.uniform(0.7, 0.99), 2)
    }