def judge_fatigue(features: dict) -> str:
    """
    根据传入的面部特征判断疲劳状态

    参数:
        features (dict): 包含面部特征的字典，包含以下字段：
            - ear: float，眼睑纵横比（Eye Aspect Ratio）
            - eye_closed_frames: int，连续闭眼的帧数
            - yawn: bool，是否检测到打哈欠

    返回:
        str: "疲劳" 或 "清醒"
    """
    ear = features.get("ear", 1.0)
    eye_closed_frames = features.get("eye_closed_frames", 0)
    yawn = features.get("yawn", False)

    # 判断规则（你可以后期微调）
    if ear < 0.2 and eye_closed_frames > 20:
        return "疲劳"
    elif yawn:
        return "疲劳"
    else:
        return "清醒"