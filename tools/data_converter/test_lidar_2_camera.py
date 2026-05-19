import numpy as np
import cv2
import open3d as o3d
import matplotlib.pyplot as plt


def project_lidar_to_image(pcd_path, img_path, annot_data):
    # 1. 解析标注数据中的内参、外参和畸变系数
    # 内参矩阵 (3x3)
    K = np.array(annot_data["cam_intrinsic"])

    # 外参矩阵 (LiDAR -> Camera, 4x4)
    extrinsic = np.array(annot_data["extrinsic"])
    R = extrinsic[:3, :3]  # 旋转矩阵
    t = extrinsic[:3, 3]  # 平移向量

    # 畸变参数 (OpenCV 8参数模型顺序: k1, k2, p1, p2, k3, k4, k5, k6)
    d = annot_data["cam_dist"]
    D = np.array([d["k1"], d["k2"], d["p1"], d["p2"],
                  d["k3"], d["k4"], d["k5"], d["k6"]], dtype=np.float64)

    # 2. 加载图像和点云
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"无法读取图像: {img_path}")
    height, width = img.shape[:2]

    pcd = o3d.io.read_point_cloud(pcd_path)
    if pcd.is_empty():
        raise ValueError(f"无法读取点云或点云为空: {pcd_path}")
    points = np.asarray(pcd.points)  # 形状为 (N, 3)

    # 3. 将点云从雷达坐标系转换到相机坐标系
    # 公式: P_cam = R * P_lidar + t
    pts_cam = (R @ points.T).T + t

    # 4. 过滤掉相机背后的点 (深度 Z <= 0)
    front_mask = pts_cam[:, 2] > 0
    pts_cam = pts_cam[front_mask]
    depths = pts_cam[:, 2]  # 保存深度信息用于后续着色

    # 5. 将相机坐标系下的 3D 点投影到 2D 图像平面
    # 因为点已经在相机坐标系下，所以 rvec 和 tvec 设为 0
    rvec = np.zeros((3, 1))
    tvec = np.zeros((3, 1))

    uv, _ = cv2.projectPoints(pts_cam, rvec, tvec, K, D)
    uv = uv.squeeze()  # 形状变为 (M, 2)

    # 6. 过滤掉超出图像边界的点
    u = np.round(uv[:, 0]).astype(int)
    v = np.round(uv[:, 1]).astype(int)

    valid_mask = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    u = u[valid_mask]
    v = v[valid_mask]
    depths = depths[valid_mask]

    # 7. 根据深度给点云着色并绘制到图像上
    # 使用 matplotlib 的 jet 颜色映射 (近处为红/橙，远处为蓝)
    norm = plt.Normalize(vmin=np.percentile(depths, 5), vmax=np.percentile(depths, 95))
    cmap = plt.cm.jet
    # cmap 返回 RGBA [0,1]，转换为 RGB [0,255]
    colors = (cmap(norm(depths))[:, :3] * 255).astype(np.uint8)

    for i in range(len(u)):
        # OpenCV 使用 BGR 格式，所以需将 RGB 转为 BGR
        r, g, b = int(colors[i][0]), int(colors[i][1]), int(colors[i][2])
        cv2.circle(img, (u[i], v[i]), radius=2, color=(b, g, r), thickness=-1)

    return img


# --- 使用示例 ---
if __name__ == "__main__":
    # 你提供的数据字典 (此处仅截取你需要用到的部分)
    annot_data = {
        "cam_intrinsic": [
            [1909.0022628665395, 0.0, 1908.9845558332952],
            [0.0, 1890.32, 1085.64],
            [0.0, 0.0, 1.0]
        ],
        "extrinsic": [
            [0.9991622623583342, -0.03919929876390687, 0.01175535858209127, 0.01679747526261534],
            [0.011193978411121776, -0.014510045885994027, -0.9998320626063748, 1.0655772457380834],
            [0.03936328652827298, 0.9991260548823679, -0.01405909417027884, -1.9262424525306043],
            [0.0, 0.0, 0.0, 1.0]
        ],
        "cam_dist": {
            "type": "pinhole",
            "k1": 6.146981432094541, "k2": 2.6485810584408953,
            "p1": -0.00014466501374551028, "p2": 0.00017586357051584356,
            "k3": 0.10268413778041566, "k4": 6.52360727904218,
            "k5": 4.769011890378588, "k6": 0.5891436486285146
        }
    }

    # 替换为你的文件路径
    PCD_FILE = "1759842903100107_merge.pcd"
    JPG_FILE = "front-camera-fov120_1759842903.100000005.jpg"
    OUTPUT_FILE = "projected_output.jpg"

    try:
        result_img = project_lidar_to_image(PCD_FILE, JPG_FILE, annot_data)

        # 保存并显示结果
        cv2.imwrite(OUTPUT_FILE, result_img)
        print(f"投影成功，已保存至: {OUTPUT_FILE}")

        # 可视化窗口显示 (按任意键关闭)
        # cv2.imshow("Projection", cv2.resize(result_img, (1280, 720)))
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    except Exception as e:
        print(f"发生错误: {e}")