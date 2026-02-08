"""
物体聚类分析脚本
功能：
1. 读取 YOLO 检测结果 (CSV)
2. 过滤高置信度数据 (>0.8)
3. 对每一类物体使用 DBSCAN 进行空间聚类 (基于3D全局坐标)
4. 计算每个聚类簇的质心和出现频次
5. 导出为大模型友好的 JSON 格式

依赖：
pip install pandas scikit-learn
"""

import pandas as pd
import numpy as np
import json
import os
import argparse
from sklearn.cluster import DBSCAN

def process_detections(csv_path, output_json_path, conf_threshold=0.8, eps=0.5, min_samples=5):
    """
    处理检测数据并生成语义地图
    
    参数:
        csv_path: 输入的CSV文件路径
        output_json_path: 输出的JSON文件路径
        conf_threshold: 置信度阈值，过滤低质量检测
        eps: DBSCAN半径 (米)，同一物体的最大距离
        min_samples: DBSCAN最小样本数，少于此数量的检测被视为噪声
    """
    
    if not os.path.exists(csv_path):
        print(f"错误：找不到输入文件 {csv_path}")
        return

    # 1. 读取数据
    print(f"正在读取数据: {csv_path}")
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"读取 CSV 失败: {e}")
        return

    # 检查必要的列是否存在
    required_cols = ['class', 'confidence', 'global_x', 'global_y', 'global_z']
    if not all(col in df.columns for col in required_cols):
        print(f"错误：CSV 文件缺少必要的列。需要: {required_cols}")
        return

    # 2. 数据清洗与预处理
    initial_count = len(df)
    # 过滤置信度
    df_filtered = df[df['confidence'] > conf_threshold].copy()
    
    # 过滤无效坐标 (例如全为0或无穷大，视具体情况而定，这里假设0,0,0可能是无效的但也可能是原点)
    # 通常深度相机无效值为0会导致坐标计算出错，但之前的代码可能已经处理过。这里做基本的非空检查。
    df_filtered = df_filtered.dropna(subset=['global_x', 'global_y', 'global_z'])
    
    print(f"数据过滤: 从 {initial_count} 条 -> {len(df_filtered)} 条 (置信度 > {conf_threshold})")
    
    if len(df_filtered) == 0:
        print("警告：过滤后没有剩余数据，请降低置信度阈值或检查数据源。")
        return

    semantic_objects_list = []

    # 3. 按类别分组聚类
    # 获取所有唯一的类别
    unique_classes = df_filtered['class'].unique()
    print(f"发现类别: {unique_classes}")

    object_id_counter = {} # 用于生成 object id, e.g., bottle_01, bottle_02

    for class_name in unique_classes:
        # 获取该类别的数据
        class_df = df_filtered[df_filtered['class'] == class_name]
        points = class_df[['global_x', 'global_y', 'global_z']].values
        
        if len(points) < min_samples:
            print(f"类别 '{class_name}' 样本数 ({len(points)}) 不足 {min_samples}，跳过聚类。")
            continue

        # 使用 DBSCAN 聚类
        # eps: 两个样本被视为邻居的最大距离 (米)
        # min_samples: 形成核心点所需的最小样本数
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
        
        labels = clustering.labels_
        
        # 统计簇的数量 (忽略噪声点 -1)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        print(f"类别 '{class_name}': 聚类出 {n_clusters} 个物体实体 (噪声点: {list(labels).count(-1)})")

        if class_name not in object_id_counter:
            object_id_counter[class_name] = 1

        # 处理每个簇
        unique_labels = set(labels)
        for label in unique_labels:
            if label == -1:
                # 噪声点，跳过
                continue
            
            # 获取属于当前簇的所有点的掩码
            class_member_mask = (labels == label)
            cluster_points = points[class_member_mask]
            
            # 4. 计算质心 (Centroid)
            # 使用均值作为该物体的最终估算坐标
            centroid = np.mean(cluster_points, axis=0) # [x, y, z]
            
            # 创建对象条目
            obj_id = f"{class_name}_{object_id_counter[class_name]:02d}"
            object_id_counter[class_name] += 1
            
            # 格式化数据，保留3位小数
            centroid_list = [round(float(x), 3) for x in centroid]
            
            semantic_obj = {
                "id": obj_id,
                "class": class_name,
                "xyz": centroid_list,
                "detection_count": int(len(cluster_points)), # 该物体被检测到的次数，作为可信度参考
                # 添加自然语言描述，便于大模型直接理解
                "description": f"位于 x={centroid_list[0]}, y={centroid_list[1]}, z={centroid_list[2]} 的 {class_name}"
            }
            
            semantic_objects_list.append(semantic_obj)

    # 5. 导出结果
    output_data = {
        "meta_info": {
            "source_file": os.path.basename(csv_path),
            "algorithm": "DBSCAN",
            "parameters": {
                "confidence_threshold": conf_threshold,
                "eps": eps,
                "min_samples": min_samples
            },
            "total_objects_found": len(semantic_objects_list)
        },
        "objects": semantic_objects_list
    }

    try:
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"\n结果已保存至: {output_json_path}")
        # 打印预览
        print("JSON 内容预览:")
        print(json.dumps(output_data, indent=2, ensure_ascii=False)[:500] + "...")
    except Exception as e:
        print(f"保存 JSON 失败: {e}")

def main():
    parser = argparse.ArgumentParser(description="对检测结果进行空间聚类，生成语义地图")
    parser.add_argument("--input", "-i", type=str, required=True, help="输入的 detection_results.csv 文件路径")
    parser.add_argument("--output", "-o", type=str, default=None, help="输出的 JSON 文件路径 (默认在输入同目录下)")
    parser.add_argument("--conf", type=float, default=0.8, help="置信度阈值 (默认 0.8)")
    parser.add_argument("--eps", type=float, default=0.5, help="聚类半径 (米) (默认 0.5)")
    parser.add_argument("--min-samples", type=int, default=5, help="聚类最小样本数 (默认 5)")

    args = parser.parse_args()

    input_path = os.path.abspath(args.input)
    if args.output:
        output_path = os.path.abspath(args.output)
    else:
        # 默认保存为同目录下 input_filename_clustered.json
        dirname = os.path.dirname(input_path)
        filename_no_ext = os.path.splitext(os.path.basename(input_path))[0]
        output_path = os.path.join(dirname, f"{filename_no_ext}_clustered.json")

    process_detections(input_path, output_path, args.conf, args.eps, args.min_samples)

if __name__ == "__main__":
    main()
