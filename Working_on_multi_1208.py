import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
import random
import os
import json
import pprint
import toml
from scipy.optimize import least_squares
from scipy.optimize import minimize
from write_to_toml_v2 import write_to_toml

# 이미지 크기 및 초기 내부 파라미터 설정
image_size = (1088.0, 1920.0)
u0 = image_size[0] / 2
v0 = image_size[1] / 2

K1 = np.array([
    [1824.6097978600892, 0.0, 1919.5],
    [0.0, 1826.6675222017589, 1079.5],
    [0.0, 0.0, 1.0]
])

K2 = np.array([
    [1824.6097978600892, 0.0, 1919.5],
    [0.0, 1826.6675222017589, 1079.5],
    [0.0, 0.0, 1.0]
])

K3 = np.array([
    [1824.6097978600892, 0.0, 1919.5],
    [0.0, 1826.6675222017589, 1079.5],
    [0.0, 0.0, 1.0]
])

K4 = np.array([
    [1824.6097978600892, 0.0, 1919.5],
    [0.0, 1826.6675222017589, 1079.5],
    [0.0, 0.0, 1.0]
])

Ks = [K1, K2, K3, K4]

###################### Data Processing ############################

def extract_high_confidence_keypoints(cam_dirs, confidence_threshold):
    """
    모든 카메라에서 주어진 confidence threshold 이상을 만족하는 키포인트만 추출.
    """
    high_confidence_keypoints = []
    camera_keypoint_counts = {}
    
    cam_files = {}
    for cam_dir in cam_dirs:
        cam_name = os.path.basename(cam_dir)
        cam_files[cam_name] = sorted([os.path.join(cam_dir, f) for f in os.listdir(cam_dir) if f.endswith('.json')])
        camera_keypoint_counts[cam_name] = 0
    
    # 모든 카메라에 대해 같은 프레임 인덱스의 파일을 동시에 로드
    for frame_files in zip(*cam_files.values()):
        frame_keypoints = {}
        
        cam_keypoints = {}
        for cam_name, frame_file in zip(cam_files.keys(), frame_files):
            with open(frame_file, 'r') as file:
                data = json.load(file)
                if data['people']:
                    keypoints = data['people'][0]['pose_keypoints_2d']
                    keypoints_conf = [(keypoints[i], keypoints[i+1], keypoints[i+2]) for i in range(0, len(keypoints), 3)]
                    cam_keypoints[cam_name] = keypoints_conf
        
        if len(cam_keypoints) > 0 and len(set(len(kp) for kp in cam_keypoints.values())) == 1:
            # 모든 카메라에서 같은 개수의 키포인트를 갖는 경우
            for i in range(len(next(iter(cam_keypoints.values())))):
                if all(cam_keypoints[cam][i][2] >= confidence_threshold for cam in cam_keypoints):
                    kp_coords = {cam: (cam_keypoints[cam][i][0], cam_keypoints[cam][i][1]) for cam in cam_keypoints}
                    frame_keypoints[i] = kp_coords
                    for cam in cam_keypoints:
                        camera_keypoint_counts[cam] += 1
        
        if frame_keypoints:
            high_confidence_keypoints.append(frame_keypoints)
    
    print("\nNumber of extracted keypoints per camera:")
    for cam_name, count in camera_keypoint_counts.items():
        print(f"{cam_name}: {count} keypoints")
    
    return high_confidence_keypoints

# 카메라 디렉토리 설정
cam_dirs = [
    r'C:\Users\gns15\OneDrive\Desktop\Calibration_with_keypoints\demo_lod\json1',
    r'C:\Users\gns15\OneDrive\Desktop\Calibration_with_keypoints\demo_lod\json2',
    r'C:\Users\gns15\OneDrive\Desktop\Calibration_with_keypoints\demo_lod\json3',
    r'C:\Users\gns15\OneDrive\Desktop\Calibration_with_keypoints\demo_lod\json4'
]

confidence_threshold = 0.55
paired_keypoints_list = extract_high_confidence_keypoints(cam_dirs, confidence_threshold)

def extract_correspondences_for_binocular(paired_keypoints_list, cam0_name="json1", cam1_name="json2"):
    pts_cam0 = []
    pts_cam1 = []
    for frame_kps in paired_keypoints_list:
        for kp_idx, cams_dict in frame_kps.items():
            if cam0_name in cams_dict and cam1_name in cams_dict:
                x0, y0 = cams_dict[cam0_name]
                x1, y1 = cams_dict[cam1_name]
                pts_cam0.append([x0, y0])
                pts_cam1.append([x1, y1])
    pts_cam0 = np.array(pts_cam0, dtype=np.float32)
    pts_cam1 = np.array(pts_cam1, dtype=np.float32)
    return pts_cam0, pts_cam1

def compute_fundamental_matrix(pts1, pts2):
    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)
    return F, mask

def compute_essential_matrix(F, K1, K2):
    return K2.T @ F @ K1

def recover_extrinsic_parameters(E, K1, pts1, pts2):
    _, R, t, mask = cv2.recoverPose(E, pts1, pts2, K1)
    return R, t, mask

def triangulate_points(K1, K2, R, t, pts_cam0, pts_cam1):
    P0 = K1 @ np.hstack((np.eye(3), np.zeros((3,1))))
    P1 = K2 @ np.hstack((R, t))
    points_4D = cv2.triangulatePoints(P0, P1, pts_cam0.T, pts_cam1.T)
    points_3D = (points_4D[:3,:] / points_4D[3,:]).T
    return points_3D

def reprojection_residual_intrinsic(intrinsic_params, pts3D, pts2D, R, t, image_size):
    fx, fy, u0, v0 = intrinsic_params
    K = np.array([[fx, 0, u0],
                  [0, fy, v0],
                  [0,  0,  1]])
    RT = np.hstack((R, t))
    pts3D_h = np.hstack([pts3D, np.ones((pts3D.shape[0],1))])
    proj = (K @ RT @ pts3D_h.T).T
    proj_2D = proj[:, :2] / proj[:, 2, None]
    residuals = (proj_2D - pts2D).ravel()
    return residuals

def optimize_intrinsic_parameters(pts3D, pts2D, R, t, K_init, image_size):
    fx_init = K_init[0,0]
    fy_init = K_init[1,1]
    u0_init = K_init[0,2]
    v0_init = K_init[1,2]
    x0 = [fx_init, fy_init, u0_init, v0_init]
    result = least_squares(reprojection_residual_intrinsic, x0, 
                           args=(pts3D, pts2D, R, t, image_size))
    fx, fy, u0, v0 = result.x
    K_optimized = np.array([[fx, 0, u0],
                            [0, fy, v0],
                            [0,  0,  1]])
    return K_optimized

def reprojection_residual_extrinsic(extrinsic_params, pts3D, pts2D, K):
    rvec = extrinsic_params[:3]
    tvec = extrinsic_params[3:]
    R, _ = cv2.Rodrigues(rvec)
    pts2D_proj, _ = cv2.projectPoints(pts3D, rvec, tvec, K, None)
    pts2D_proj = pts2D_proj.reshape(-1,2)
    residuals = (pts2D_proj - pts2D).ravel()
    return residuals

def optimize_extrinsic_parameters(pts3D, pts2D, K, R_init, t_init):
    rvec, _ = cv2.Rodrigues(R_init)
    x0 = np.hstack((rvec.ravel(), t_init.ravel()))
    result = least_squares(reprojection_residual_extrinsic, x0, args=(pts3D, pts2D, K))
    rvec_opt = result.x[:3]
    t_opt = result.x[3:]
    R_opt, _ = cv2.Rodrigues(rvec_opt)
    t_opt = t_opt.reshape(3,1)
    return R_opt, t_opt

def compute_reprojection_error(pts3D, pts2D, R, t, K):
    pts3D_h = np.hstack([pts3D, np.ones((pts3D.shape[0],1))])
    RT = np.hstack((R, t))
    proj = (K @ RT @ pts3D_h.T).T
    proj_2D = proj[:, :2] / proj[:, 2, None]
    errors = np.sqrt(np.sum((proj_2D - pts2D)**2, axis=1))
    return np.mean(errors)

def joint_optimization(inlier_pts_cam0, inlier_pts_cam1, K0, K, R, t, image_size, max_iter=10):
    for i in range(max_iter):
        pts3D = triangulate_points(K0, K, R, t, inlier_pts_cam0, inlier_pts_cam1)
        err_before = compute_reprojection_error(pts3D, inlier_pts_cam1, R, t, K)
        print(f"Iteration {i} - Before extrinsic optimization: Reprojection error = {err_before:.4f}")
        
        R, t = optimize_extrinsic_parameters(pts3D, inlier_pts_cam1, K, R, t)
        
        pts3D = triangulate_points(K0, K, R, t, inlier_pts_cam0, inlier_pts_cam1)
        err_after_ext = compute_reprojection_error(pts3D, inlier_pts_cam1, R, t, K)
        print(f"Iteration {i} - After extrinsic optimization: Reprojection error = {err_after_ext:.4f}")
        
        K = optimize_intrinsic_parameters(pts3D, inlier_pts_cam1, R, t, K, image_size)
        
        pts3D = triangulate_points(K0, K, R, t, inlier_pts_cam0, inlier_pts_cam1)
        err_after_int = compute_reprojection_error(pts3D, inlier_pts_cam1, R, t, K)
        print(f"Iteration {i} - After intrinsic optimization: Reprojection error = {err_after_int:.4f}\n")
        
    return K, R, t

# 파이프라인 수행
pts_cam0, pts_cam1 = extract_correspondences_for_binocular(paired_keypoints_list, "json1", "json2")
F, mask = compute_fundamental_matrix(pts_cam0, pts_cam1)
inlier_pts_cam0 = pts_cam0[mask.ravel()==1]
inlier_pts_cam1 = pts_cam1[mask.ravel()==1]

K0 = Ks[0]
K1_ = Ks[1]

E = compute_essential_matrix(F, K0, K1_)
R, t, _ = recover_extrinsic_parameters(E, K0, inlier_pts_cam0, inlier_pts_cam1)

points_3D = triangulate_points(K0, K1_, R, t, inlier_pts_cam0, inlier_pts_cam1)
K1_optimized = optimize_intrinsic_parameters(points_3D, inlier_pts_cam1, R, t, K1_, image_size)
K1_final, R_final, t_final = joint_optimization(inlier_pts_cam0, inlier_pts_cam1, K0, K1_optimized, R, t, image_size)

print("Camera 1 final parameters:")
print("K1_final:\n", K1_final)
print("R_final:\n", R_final)
print("t_final:\n", t_final)

###################### Additional Code for (3) Parameter initialization for other cameras ############################
def extract_2D_points_for_camera_i(paired_keypoints_list, cam_name, inlier_pts_cam0, inlier_pts_cam1):
    pts2D_i = []
    # 매칭 시 cam0,cam1에 해당하는 점과 동일한 점을 가진 frame을 찾아 cam i 포인트 추출
    # 약간의 오차 허용을 위해 소수점 비교 시 threshold 사용
    eps = 1e-3
    # inlier_pts_cam0, inlier_pts_cam1 길이만큼 반복
    for (x0, y0), (x1, y1) in zip(inlier_pts_cam0, inlier_pts_cam1):
        found = False
        for frame_kps in paired_keypoints_list:
            for kp_idx, cams_dict in frame_kps.items():
                if ("json1" in cams_dict and "json2" in cams_dict):
                    c0 = cams_dict["json1"]
                    c1 = cams_dict["json2"]
                    if abs(c0[0]-x0)<eps and abs(c0[1]-y0)<eps and abs(c1[0]-x1)<eps and abs(c1[1]-y1)<eps:
                        if cam_name in cams_dict:
                            pts2D_i.append(cams_dict[cam_name])
                            found = True
                            break
            if found:
                break
    pts2D_i = np.array(pts2D_i, dtype=np.float32)
    return pts2D_i

def initialize_camera_i(i, cam_name, K0, K_ref, R_ref, t_ref, points_3D, inlier_pts_cam0, inlier_pts_cam1):
    """
    camera i를 초기화하는 함수.
    여기서는 간소하게 camera i에 대한 pts2D를 추출 후 solvePnP 적용.
    """
    K_i_init = Ks[i]
    pts2D_i = extract_2D_points_for_camera_i(paired_keypoints_list, cam_name, inlier_pts_cam0, inlier_pts_cam1)
    print(f"Extracted {pts2D_i.shape[0]} points for camera {i} ({cam_name})")
    
    
    if pts2D_i.shape[0] < 6:
        print(f"Not enough points to initialize camera {i} ({cam_name})")
        return K_i_init, np.eye(3), np.zeros((3,1))
    
    retval, rvec_i, tvec_i = cv2.solvePnP(points_3D, pts2D_i, K_i_init, None)
    R_i, _ = cv2.Rodrigues(rvec_i)
    t_i = tvec_i
    print(f"Initial extrinsic parameters for camera {i} ({cam_name}):")
    print("R_i:\n", R_i)
    print("t_i:\n", t_i)
    
    # Extrinsic 최적화
    R_i, t_i = optimize_extrinsic_parameters(points_3D, pts2D_i, K_i_init, R_i, t_i)
    
    # Intrinsic 최적화
    K_i_optimized = optimize_intrinsic_parameters(points_3D, pts2D_i, R_i, t_i, K_i_init, image_size)
    
    return K_i_optimized, R_i, t_i

# Camera0,1은 이미 초기화 끝남 (K0, K1_final, R_final, t_final)
# Camera2,3 초기화
cam2_name = "json3"
cam3_name = "json4"

K2_final, R2_final, t2_final = initialize_camera_i(2, cam2_name, K0, K1_final, R_final, t_final, points_3D, inlier_pts_cam0, inlier_pts_cam1)
K3_final, R3_final, t3_final = initialize_camera_i(3, cam3_name, K0, K1_final, R_final, t_final, points_3D, inlier_pts_cam0, inlier_pts_cam1)

print("Camera 2 initialization done:")
print("K2:\n", K2_final)
print("R2:\n", R2_final)
print("t2:\n", t2_final)

print("Camera 3 initialization done:")
print("K3:\n", K3_final)
print("R3:\n", R3_final)
print("t3:\n", t3_final)

print("All cameras initialized.")
