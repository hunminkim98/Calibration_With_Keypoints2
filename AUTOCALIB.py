import os
import json
import numpy as np
import cv2
from scipy.optimize import least_squares
import toml

# --- Constants ---
IMAGE_SIZE = (3840.0, 2160.0)  # 이미지 크기
U0 = IMAGE_SIZE[0] / 2  # 주점 u0
V0 = IMAGE_SIZE[1] / 2  # 주점 v0
CONFIDENCE_THRESHOLD = 0.6  # 신뢰도 임계값
MAX_ITERATIONS = 5  # 공동 최적화 반복 횟수

# 초기 내부 파라미터 (카메라별로 동일하다고 가정)
INITIAL_K = np.array([
    [1824.6097978600892, 0.0, U0],
    [0.0, 1826.6675222017589, V0],
    [0.0, 0.0, 1.0]
])

# --- Helper Functions ---
def load_keypoints_from_json(cam_dirs, confidence_threshold):
    """JSON 파일에서 유효한 2D 관절을 로드합니다."""
    all_cam_data = []
    for cam_dir in cam_dirs:
        cam_files = sorted([os.path.join(cam_dir, f) for f in os.listdir(cam_dir) if f.endswith('.json')])
        cam_data = []
        for cam_file in cam_files:
            with open(cam_file, 'r') as file:
                data = json.load(file)
                try:
                    keypoints = data['people'][0]['pose_keypoints_2d']
                    keypoints_conf = [(keypoints[i], keypoints[i+1], keypoints[i+2]) for i in range(0, len(keypoints), 3)]
                    valid_keypoints = [(x, y) for x, y, conf in keypoints_conf if conf >= confidence_threshold]
                    cam_data.append(valid_keypoints)
                except:
                    continue
        all_cam_data.append(cam_data)
    return all_cam_data

def extract_paired_keypoints(ref_cam_data, other_cam_data):
    """참조 카메라와 다른 카메라 간의 paired keypoints를 추출합니다. 두 카메라의 관절 개수가 동일한 프레임만 사용."""
    paired_keypoints = []
    for ref_frame, other_frame in zip(ref_cam_data, other_cam_data):
        if len(ref_frame) == len(other_frame) and len(ref_frame) > 0:
            paired_keypoints.append(list(zip(ref_frame, other_frame)))
    return paired_keypoints

def compute_fundamental_matrix(points1, points2):
    """RANSAC을 사용하여 기본 행렬을 계산합니다."""
    points1 = np.array(points1, dtype=float).reshape(-1, 2)
    points2 = np.array(points2, dtype=float).reshape(-1, 2)
    F, _ = cv2.findFundamentalMat(points1, points2, cv2.FM_RANSAC)
    return F

def compute_essential_matrix(F, K1, K2):
    """본질 행렬을 계산합니다."""
    return K2.T @ F @ K1

def recover_pose_from_essential_matrix(E, points1, points2, K):
    """본질 행렬에서 R과 t를 추출합니다."""
    points1 = np.array(points1, dtype=float).reshape(-1, 2)
    points2 = np.array(points2, dtype=float).reshape(-1, 2)
    _, R, t, _ = cv2.recoverPose(E, points1, points2, K)
    return R, t

def triangulate_points(paired_keypoints, P1, P2):
    """paired keypoints를 사용하여 3D 점을 삼각측량합니다."""
    points_3d = []
    for frame in paired_keypoints:
        points1, points2 = zip(*frame)
        points1 = np.array(points1, dtype=float).T
        points2 = np.array(points2, dtype=float).T
        points_4d = cv2.triangulatePoints(P1, P2, points1, points2)
        points_3d_frame = points_4d[:3] / points_4d[3]
        points_3d.append(points_3d_frame.T)
    return points_3d

def compute_reprojection_error(points_3d, points_2d, K, R, t):
    """재투영 오차를 계산합니다. 프레임별 점 개수가 일치하지 않으면 건너뜁니다."""
    total_error = 0
    total_points = 0
    RT = np.hstack((R, t.reshape(-1, 1)))
    P = K @ RT
    for frame_3d, frame_2d in zip(points_3d, points_2d):
        if len(frame_3d) != len(frame_2d):
            continue  # 개수가 일치하지 않으면 건너뜀
        points_3d_hom = np.hstack((frame_3d, np.ones((frame_3d.shape[0], 1))))
        points_2d_proj = (P @ points_3d_hom.T).T
        points_2d_proj = points_2d_proj[:, :2] / points_2d_proj[:, 2:]
        error = np.linalg.norm(points_2d_proj - np.array(frame_2d), axis=1)
        total_error += np.sum(error)
        total_points += len(frame_2d)
    return total_error / total_points if total_points > 0 else 0

# --- Optimization Functions ---
def vectorize_for_optimization(points_3d, points_2d):
    """최적화를 위해 데이터를 벡터화합니다."""
    u_detected = []
    v_detected = []
    X_w = []
    Y_w = []
    Z_w = []
    for frame_3d, frame_2d in zip(points_3d, points_2d):
        if len(frame_3d) != len(frame_2d):
            continue
        for point_3d, point_2d in zip(frame_3d, frame_2d):
            u_detected.append(point_2d[0])
            v_detected.append(point_2d[1])
            X_w.append(point_3d[0])
            Y_w.append(point_3d[1])
            Z_w.append(point_3d[2])
    return np.array(u_detected), np.array(v_detected), np.array(X_w), np.array(Y_w), np.array(Z_w)

def intrinsic_loss(x, u_detected, v_detected, X_c, Y_c, Z_c, u0, v0):
    """내부 파라미터 최적화를 위한 손실 함수"""
    f_x, f_y = x
    loss_u = Z_c * u_detected - (f_x * X_c + u0 * Z_c)
    loss_v = Z_c * v_detected - (f_y * Y_c + v0 * Z_c)
    return np.concatenate([loss_u, loss_v])

def extrinsic_loss(x, K, u_detected, v_detected, X_w, Y_w, Z_w, is_first_other_camera):
    """외부 파라미터 최적화를 위한 손실 함수"""
    t = x
    R = np.eye(3)  # R은 고정 (카메라 간 상대적 회전은 미리 계산됨)
    RT = np.hstack((R, t.reshape(-1, 1)))
    P = K @ RT
    points_3d_hom = np.vstack([X_w, Y_w, Z_w, np.ones_like(X_w)])
    points_2d_proj = P @ points_3d_hom
    points_2d_proj = points_2d_proj[:2] / points_2d_proj[2]
    loss_u = points_2d_proj[0] - u_detected
    loss_v = points_2d_proj[1] - v_detected
    loss = np.concatenate([loss_u, loss_v])
    if is_first_other_camera:
        t_norm = np.linalg.norm(t)
        penalty = (t_norm - 1) ** 2
        loss = np.append(loss, penalty * 100)  # 패널티 가중치
    return loss

def optimize_intrinsic(points_3d, points_2d, K, R, t):
    """내부 파라미터를 최적화합니다."""
    u_detected, v_detected, X_w, Y_w, Z_w = vectorize_for_optimization(points_3d, points_2d)
    RT = np.hstack((R, t.reshape(-1, 1)))
    points_cam = RT @ np.vstack([X_w, Y_w, Z_w, np.ones_like(X_w)])
    X_c, Y_c, Z_c = points_cam[:3]
    x0 = [K[0, 0], K[1, 1]]
    result = least_squares(intrinsic_loss, x0, args=(u_detected, v_detected, X_c, Y_c, Z_c, U0, V0), method='trf')
    K_optimized = np.array([[result.x[0], 0, U0], [0, result.x[1], V0], [0, 0, 1]])
    return K_optimized

def optimize_extrinsic(points_3d, points_2d, K, R, t, is_first_other_camera):
    """외부 파라미터를 최적화합니다."""
    u_detected, v_detected, X_w, Y_w, Z_w = vectorize_for_optimization(points_3d, points_2d)
    x0 = t.flatten()
    result = least_squares(extrinsic_loss, x0, args=(K, u_detected, v_detected, X_w, Y_w, Z_w, is_first_other_camera), method='trf')
    t_optimized = result.x.reshape(-1, 1)
    return t_optimized

# --- Main Calibration Function ---
def calibrate_multi_camera(cam_dirs, ref_cam_idx):
    """멀티카메라 시스템을 보정합니다."""
    all_cam_data = load_keypoints_from_json(cam_dirs, CONFIDENCE_THRESHOLD)
    num_cams = len(cam_dirs)
    K_list = [INITIAL_K.copy() for _ in range(num_cams)]
    R_list = [np.eye(3) for _ in range(num_cams)]
    t_list = [np.zeros((3, 1)) for _ in range(num_cams)]
    first_other_camera = True

    for i in range(num_cams):
        if i == ref_cam_idx:
            continue
        print(f"Calibrating camera {i} with reference camera {ref_cam_idx}...")
        paired_keypoints = extract_paired_keypoints(all_cam_data[ref_cam_idx], all_cam_data[i])
        points1 = [p[0] for frame in paired_keypoints for p in frame]
        points2 = [p[1] for frame in paired_keypoints for p in frame]
        F = compute_fundamental_matrix(points1, points2)
        E = compute_essential_matrix(F, K_list[ref_cam_idx], K_list[i])
        R, t = recover_pose_from_essential_matrix(E, points1, points2, K_list[i])
        R_list[i] = R
        t_list[i] = t

        for _ in range(MAX_ITERATIONS):
            P1 = K_list[ref_cam_idx] @ np.hstack((np.eye(3), np.zeros((3, 1))))
            P2 = K_list[i] @ np.hstack((R_list[i], t_list[i]))
            points_3d = triangulate_points(paired_keypoints, P1, P2)
            t_list[i] = optimize_extrinsic(points_3d, all_cam_data[i], K_list[i], R_list[i], t_list[i], first_other_camera)
            K_list[i] = optimize_intrinsic(points_3d, all_cam_data[i], K_list[i], R_list[i], t_list[i])
            first_other_camera = False  # 첫 번째 카메라 이후에는 False

        error = compute_reprojection_error(points_3d, all_cam_data[i], K_list[i], R_list[i], t_list[i])
        print(f"Camera {i} calibrated with reprojection error: {error}")

    return K_list, R_list, t_list

# --- Save Results ---
def save_to_toml(K_list, R_list, t_list, cam_dirs, output_file):
    """보정 결과를 사용자 지정 TOML 형식으로 저장합니다."""
    data = {}
    total_error = 0.0

    for i, cam_dir in enumerate(cam_dirs):
        cam_name = f"int_cam{i}_img"
        R_rod, _ = cv2.Rodrigues(R_list[i])
        R_rod = R_rod.flatten().tolist()
        t_flat = t_list[i].flatten().tolist()
        data[cam_name] = {
            "name": cam_name,
            "size": [3840.0, 2160.0],
            "matrix": K_list[i].tolist(),
            "distortions": [0.0, 0.0, 0.0, 0.0],
            "rotation": R_rod,
            "translation": t_flat,
            "fisheye": False
        }
        if i != ref_cam_idx:
            paired_keypoints = extract_paired_keypoints(
                load_keypoints_from_json([cam_dirs[ref_cam_idx]], CONFIDENCE_THRESHOLD)[0],
                load_keypoints_from_json([cam_dir], CONFIDENCE_THRESHOLD)[0]
            )
            P1 = K_list[ref_cam_idx] @ np.hstack((np.eye(3), np.zeros((3, 1))))
            P2 = K_list[i] @ np.hstack((R_list[i], t_list[i]))
            points_3d = triangulate_points(paired_keypoints, P1, P2)
            error = compute_reprojection_error(points_3d, load_keypoints_from_json([cam_dir], CONFIDENCE_THRESHOLD)[0], K_list[i], R_list[i], t_list[i])
            total_error += error

    data["metadata"] = {
        "adjusted": False,
        "error": total_error / (len(cam_dirs) - 1) if len(cam_dirs) > 1 else 0.0
    }

    with open(output_file, 'w') as f:
        toml.dump(data, f)

# --- Main Execution ---
if __name__ == "__main__":
    cam_dirs = [
        r'D:\calibration\Calibration_with_keypoints\cal_json1',
        r'D:\calibration\Calibration_with_keypoints\cal_json2',
        r'D:\calibration\Calibration_with_keypoints\cal_json3',
        r'D:\calibration\Calibration_with_keypoints\cal_json4'
    ]
    ref_cam_idx = 0  # 예시로 첫 번째 카메라를 참조로 설정
    K_list, R_list, t_list = calibrate_multi_camera(cam_dirs, ref_cam_idx)
    save_to_toml(K_list, R_list, t_list, cam_dirs, 'calibration_results.toml')