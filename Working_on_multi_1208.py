import numpy as np
import cv2
import os
import json
from scipy.optimize import least_squares

image_size = (1088.0, 1920.0)
Ks = [np.array([[1824.6, 0.0, 1919.5],
                [0.0, 1826.7, 1079.5],
                [0.0, 0.0, 1.0]]) for _ in range(4)]
image_size = (1088.0, 1920.0)
u0 = image_size[0]/2
v0 = image_size[1]/2
f_init = 1824.6

# K를 f만 변수로 하고 u0,v0 고정
def K_from_f(f):
    return np.array([[f,0,u0],[0,f,v0],[0,0,1]])

def extract_high_confidence_keypoints(cam_dirs, confidence_threshold):
    high_confidence_keypoints = []
    camera_keypoint_counts = {}
    cam_files = {}
    for cam_dir in cam_dirs:
        cam_name = os.path.basename(cam_dir)
        cam_files[cam_name] = sorted([os.path.join(cam_dir,f) for f in os.listdir(cam_dir) if f.endswith('.json')])
        camera_keypoint_counts[cam_name] = 0

    for frame_files in zip(*cam_files.values()):
        frame_keypoints = {}
        cam_keypoints = {}
        for cam_name, frame_file in zip(cam_files.keys(), frame_files):
            with open(frame_file, 'r') as file:
                data = json.load(file)
                if data['people']:
                    k = data['people'][0]['pose_keypoints_2d']
                    keypoints_conf = [(k[i], k[i+1], k[i+2]) for i in range(0,len(k),3)]
                    cam_keypoints[cam_name] = keypoints_conf

        if cam_keypoints and len(set(len(kp) for kp in cam_keypoints.values()))==1:
            N = len(next(iter(cam_keypoints.values())))
            for i in range(N):
                if all(cam_keypoints[cam][i][2]>=confidence_threshold for cam in cam_keypoints):
                    coords = {cam:(cam_keypoints[cam][i][0], cam_keypoints[cam][i][1]) for cam in cam_keypoints}
                    frame_keypoints[i] = coords
                    for cam in cam_keypoints:
                        camera_keypoint_counts[cam]+=1
        if frame_keypoints:
            high_confidence_keypoints.append(frame_keypoints)

    print("\n[INFO] Number of extracted keypoints per camera:")
    for cam_name, count in camera_keypoint_counts.items():
        print(f"{cam_name}: {count} keypoints")
    return high_confidence_keypoints

def compute_fundamental_matrix(pts1,pts2):
    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)
    return F, mask

def compute_essential_matrix(F,K1,K2): 
    return K2.T@F@K1

def recover_extrinsic_parameters(E,K1,pts1,pts2):
    _, R,t,mask = cv2.recoverPose(E,pts1,pts2,K1)
    return R,t,mask

def triangulate_points(K1,K2,R,t,pts0,pts1):
    P0=K1@np.hstack((np.eye(3),np.zeros((3,1))))
    P1=K2@np.hstack((R,t))
    X=cv2.triangulatePoints(P0,P1,pts0.T,pts1.T)
    return (X[:3]/X[3]).T

# eq(5) 기반 residual: f만 변수, u0,v0고정
def residual_intrinsic_eq5(f, pts3D, pts2D, R, t, img_size):
    K = K_from_f(f[0])
    RT = np.hstack((R,t))
    P = np.vstack((RT,[0,0,0,1]))
    ptsW_h = np.hstack([pts3D, np.ones((len(pts3D),1))])
    ptsC = (P@ptsW_h.T).T
    Xc=ptsC[:,0]; Yc=ptsC[:,1]; Zc=ptsC[:,2]
    u=pts2D[:,0]; v=pts2D[:,1]

    # diff_u = Zc*u - (f*Xc + u0*Zc)
    # diff_v = Zc*v - (f*Yc + v0*Zc)
    diff_u = Zc*u - (f[0]*Xc + u0*Zc)
    diff_v = Zc*v - (f[0]*Yc + v0*Zc)

    return np.hstack([diff_u, diff_v])

def optimize_intrinsic_f(pts3D,pts2D,R,t,f_init,img_size):
    # f만 최적화
    res=least_squares(residual_intrinsic_eq5,[f_init],args=(pts3D,pts2D,R,t,img_size))
    f_opt=res.x[0]
    return f_opt

def residual_extrinsic_eq9(tvec, pts3D, pts2D, f, R_fixed, cam_id, h):
    # R_fixed 고정
    K=K_from_f(f)
    rvec,_=cv2.Rodrigues(R_fixed)
    pts2D_proj,_=cv2.projectPoints(pts3D,rvec,tvec,K,None)
    pts2D_proj=pts2D_proj.reshape(-1,2)
    residuals = (pts2D_proj - pts2D).ravel()
    if cam_id == 1:
        norm_t = np.linalg.norm(tvec)
        penalty = h*(norm_t-1.0)
        residuals = np.append(residuals, penalty)
    return residuals

def optimize_extrinsic_eq9(pts3D,pts2D,f,R_fixed,t_init,cam_id,h):
    # t만 최적화, R_fixed, f고정
    res=least_squares(residual_extrinsic_eq9,t_init.ravel(),args=(pts3D,pts2D,f,R_fixed,cam_id,h))
    t_opt=res.x.reshape(3,1)
    return t_opt

def compute_error(pts3D,pts2D,R,t,f):
    K=K_from_f(f)
    Xh=np.hstack([pts3D,np.ones((len(pts3D),1))])
    RT=np.hstack((R,t))
    proj=(K@RT@Xh.T).T
    proj2D=proj[:,:2]/proj[:,2,None]
    err=np.sqrt(np.sum((proj2D-pts2D)**2,axis=1))
    return np.mean(err)

def extract_points_for_two_cameras(paired_keypoints_list, camA, camB):
    ptsA, ptsB = [], []
    for frame_kps in paired_keypoints_list:
        for kp_idx, cams_dict in frame_kps.items():
            if camA in cams_dict and camB in cams_dict:
                ptsA.append(cams_dict[camA])
                ptsB.append(cams_dict[camB])
    return np.array(ptsA,np.float32), np.array(ptsB,np.float32)

def extract_points_for_single_camera(paired_keypoints_list, reference_pts0, reference_pts1, cam_name):
    pts_i = []
    for frame_kps in paired_keypoints_list:
        for kp_idx, cams_dict in frame_kps.items():
            if ("json1" in cams_dict and "json2" in cams_dict and cam_name in cams_dict):
                pts_i.append(cams_dict[cam_name])
    min_len = min(len(reference_pts0), len(pts_i))
    pts_i = pts_i[:min_len]
    return np.array(pts_i, np.float32)

def initialize_camera_i(i, cam_name, paired_keypoints_list, pts_cam0, pts_cam1, R_fixed, t_fixed, points_3D, f_init):
    # camera i initialization: solvePnP -> extrinsic(opt t) -> intrinsic(opt f)
    print(f"\n[INFO] Initializing camera {i} ({cam_name})...")
    pts2D_i = extract_points_for_single_camera(paired_keypoints_list, pts_cam0, pts_cam1, cam_name)
    if len(pts2D_i)<6:
        print(f"[WARN] Not enough points for camera {i} ({cam_name})")
        return f_init, R_fixed, t_fixed[i], pts2D_i, points_3D
    print(f"[DEBUG] Extracted {len(pts2D_i)} points for camera {i}")

    min_len = min(len(points_3D), len(pts2D_i))
    pts3D = points_3D[:min_len]
    pts2D_i = pts2D_i[:min_len]

    # 초기 extrinsic: 회전은 고정(초기 R), 번역 solvePnP로 초기화
    # 여기서는 회전을 초기 R_fixed[i]로부터 가져오지 않고 solvePnP로 얻은 R_i를 무시하고 R_fixed 사용
    # 왜냐하면 R 고정 필요. 여기서는 R_fixed[i]를 초기 R이라 가정.
    # 첫 camera0: R=I, t=0, 나머지도 초기 설정 필요(여기서는 임의로 첫 initialization 후 R고정)
    if i==0:
        # 기준 카메라 t=0, R=I
        R_i = R_fixed[i]
        t_i = t_fixed[i]
    else:
        # solvePnP
        retval,rvec_i,tvec_i=cv2.solvePnP(pts3D,pts2D_i,K_from_f(f_init),None)
        R_i,_=cv2.Rodrigues(rvec_i)
        # R고정: R_i <- R_fixed[i]
        R_i=R_fixed[i]
        t_i=tvec_i

    print(f"[DEBUG] Initial R,t for camera {i}:\nR:\n{R_i}\nt:\n{t_i}")

    h=10.0
    if i==0:
        # ref cam: no extrinsic update(t=0)
        pass
    elif i==1:
        t_i=optimize_extrinsic_eq9(pts3D,pts2D_i,f_init,R_i,t_i,cam_id=1,h=h)
    else:
        t_i=optimize_extrinsic_eq9(pts3D,pts2D_i,f_init,R_i,t_i,cam_id=i,h=0.0)

    f_opt=optimize_intrinsic_f(pts3D,pts2D_i,R_i,t_i,f_init,image_size)
    print(f"[INFO] Camera {i} ({cam_name}) initialization done.")
    return f_opt,R_i,t_i,pts2D_i,pts3D

#############################################
# 메인 실행부
cam_dirs = [
    r'C:\Users\gns15\OneDrive\Desktop\Calibration_with_keypoints\cal_json1',
    r'C:\Users\gns15\OneDrive\Desktop\Calibration_with_keypoints\cal_json2',
    r'C:\Users\gns15\OneDrive\Desktop\Calibration_with_keypoints\cal_json3',
    r'C:\Users\gns15\OneDrive\Desktop\Calibration_with_keypoints\cal_json4'
]

paired_keypoints_list=extract_high_confidence_keypoints(cam_dirs,0.55)

pts_cam0, pts_cam1 = extract_points_for_two_cameras(paired_keypoints_list,"json1","json2")
F,mask=compute_fundamental_matrix(pts_cam0,pts_cam1)

# 초기 R,t
R_init=np.eye(3)
t_init=np.zeros((3,1))
K0=K_from_f(f_init)
K1=K_from_f(f_init)

E=compute_essential_matrix(F,K0,K1)
R,t,_=recover_extrinsic_parameters(E,K0,pts_cam0,pts_cam1)
# 기준: camera0: R0=I, t0=0
R_fixed=[np.eye(3), R, R, R] # 초기 R들(추정후 고정)
t_fixed=[np.zeros((3,1)), t, np.zeros((3,1)), np.zeros((3,1))] # cam1은 t로 시작, cam2,3 나중에 solvePnP후 업데이트

points_3D=triangulate_points(K0,K1,R,t,pts_cam0,pts_cam1)

# camera0은 ref, f_init 그대로, R0=I,t0=0
f_list=[f_init, f_init, f_init, f_init]
f_list[0]=f_init
# cam1 init
f_list[1],R_fixed[1],t_fixed[1],pts2D_cam1_,pts3D_cam1_=initialize_camera_i(1,"json2",paired_keypoints_list,pts_cam0,pts_cam1,R_fixed,t_fixed,points_3D,f_init)
# cam2 init
f_list[2],R_fixed[2],t_fixed[2],pts2D_cam2,pts3D_cam2=initialize_camera_i(2,"json3",paired_keypoints_list,pts_cam0,pts_cam1,R_fixed,t_fixed,points_3D,f_init)
# cam3 init
f_list[3],R_fixed[3],t_fixed[3],pts2D_cam3,pts3D_cam3=initialize_camera_i(3,"json4",paired_keypoints_list,pts_cam0,pts_cam1,R_fixed,t_fixed,points_3D,f_init)

print("All cameras initialized.")

# 이후 모든 카메라 Extrinsic, Intrinsic 순차 최적화 루프
# Extrinsic: cam1에 penalty, cam2, cam3 no penalty, cam0고정
# Intrinsic: 모든 cam f 최적화
def extrinsic_all(pts0,pts1,pts2D_cams, R_fixed, t_fixed, f_list, h=10.0):
    # 모든 카메라 번역 최적화
    pts3D=triangulate_points(K_from_f(f_list[0]),K_from_f(f_list[1]),R_fixed[1],t_fixed[1],pts0,pts1)
    # cam1
    t_fixed[1]=optimize_extrinsic_eq9(pts3D,pts2D_cams[1],f_list[1],R_fixed[1],t_fixed[1],1,h)
    # cam2
    min2 = min(len(pts3D), len(pts2D_cams[2]))
    t_fixed[2]=optimize_extrinsic_eq9(pts3D[:min2],pts2D_cams[2][:min2],f_list[2],R_fixed[2],t_fixed[2],2,0.0)
    # cam3
    min3 = min(len(pts3D), len(pts2D_cams[3]))
    t_fixed[3]=optimize_extrinsic_eq9(pts3D[:min3],pts2D_cams[3][:min3],f_list[3],R_fixed[3],t_fixed[3],3,0.0)
    return t_fixed

def intrinsic_all(pts0,pts1,pts2D_cams,R_fixed,t_fixed,f_list):
    pts3D=triangulate_points(K_from_f(f_list[0]),K_from_f(f_list[1]),R_fixed[1],t_fixed[1],pts0,pts1)
    for cid in [0,1,2,3]:
        min_len = min(len(pts3D), len(pts2D_cams[cid]))
        f_list[cid]=optimize_intrinsic_f(pts3D[:min_len],pts2D_cams[cid][:min_len],R_fixed[cid],t_fixed[cid],f_list[cid],image_size)
    return f_list

pts2D_cams = {
    0: pts_cam0[:len(points_3D)],
    1: pts_cam1[:len(points_3D)],
    2: pts2D_cam2[:len(points_3D)],
    3: pts2D_cam3[:len(points_3D)]
}

print("\n[INFO] Full multi-camera optimization loop:")
for i in range(5):
    t_fixed=extrinsic_all(pts_cam0,pts_cam1,pts2D_cams,R_fixed,t_fixed,f_list,h=10.0)
    f_list=intrinsic_all(pts_cam0,pts_cam1,pts2D_cams,R_fixed,t_fixed,f_list)
    pts3D_all=triangulate_points(K_from_f(f_list[0]),K_from_f(f_list[1]),R_fixed[1],t_fixed[1],pts_cam0,pts_cam1)
    err=compute_error(pts3D_all,pts_cam1,R_fixed[1],t_fixed[1],f_list[1])
    print(f"Iteration {i}, avg error: {err:.4f}")

print("All cameras fully optimized with fixed principal point, fixed rotation, fixed ref cam at origin, and only f optimized according to eq(5) and eq(9).")


print("\n[FINAL RESULTS] Camera parameters after full optimization:")
for cam_idx in range(4):
    print(f"\nCamera {cam_idx} (json{cam_idx+1}):")
    print("Intrinsic matrix K:")
    print(Ks[cam_idx])
    if cam_idx == 0:
        print("Extrinsic parameters (Reference camera):")
        print("R:\n", np.eye(3))
        print("t:\n", np.zeros((3,1)))
    else:
        print("Extrinsic parameters:")
        print("R:\n", R if cam_idx == 1 else R)  # All cameras share same R relative to cam0
        print("t:\n", t_list[cam_idx])
    
print("\nReprojection errors:")
errors = {0: None}  # Reference camera has no error
for cam_idx in range(4):  # We have 4 cameras (0-3)
    if cam_idx == 0:
        continue  # Skip reference camera
    pts3D = triangulate_points(K0, Ks[cam_idx], R, t_list[cam_idx], pts_cam0, pts2D_list[cam_idx])
    err = compute_error(pts3D, pts2D_list[cam_idx], R, t_list[cam_idx], Ks[cam_idx])
    errors[cam_idx] = err
    print(f"Camera {cam_idx} (json{cam_idx+1}): {err:.4f} pixels")

print("\nAll cameras fully optimized with eq(5) and eq(9).")

# Save calibration results to TOML file
from write_to_toml_v2 import write_calib_to_toml
write_calib_to_toml(Ks, R, t_list, errors, image_size=image_size, output_file="Calib.toml")
print("\nCalibration results saved to Calib.toml")
