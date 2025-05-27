# test_demo.py
from fatigue_detector_KNN_class import FatigueDetector
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pprint import pprint
import warnings

warnings.filterwarnings("ignore")
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

model = FatigueDetector()

def risk_level(prob):
    if prob < 0.4:
        return "低"
    elif prob < 0.7:
        return "中"
    else:
        return "高"

def generate_report(sample_id, sample, result):
    # 假设原始标签为中文"疲劳"和"清醒
    fatigue_prob = result.get("概率", {}).get("疲劳", 0)    #get概率，再get概率里的疲劳概率
    
    return {
        "疲劳状态": result["状态"],
        "疲劳风险等级": risk_level(fatigue_prob)
    }

#绘图
def plot_fatigue_analysis(reports):
    df = pd.DataFrame(reports)

    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x='疲劳风险等级', order=["低", "中", "高"])
    plt.title("疲劳风险等级分布")
    plt.show()

    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x='疲劳状态')
    plt.title("疲劳状态分布")
    plt.show()


from jiekou import extract_fatigue_features  # 确保你有这个模块

def process_images(image_paths):
    if isinstance(image_paths, str):
        image_paths = [image_paths]

    reports = []
    for i, path in enumerate(image_paths, 1):
        features = extract_fatigue_features(path)
        result = model.judge(features)
        report = generate_report(i, features, result)
        reports.append(report)

    return reports[0] if len(reports) == 1 else reports

image_path = './images/1.jpg'
print("单张图片分析结果：")
image_list = ['./images/1.jpg', './images/2.jpg']
print("\n多张图片分析统计图：")
reports = process_images(image_list)
plot_fatigue_analysis(reports)
pprint(reports)
