import numpy as np
import cv2
import os
import json
from scipy.optimize import least_squares

image_size = (2704.0, 2028.0)
u0 = image_size[0] / 2
v0 = image_size[1] / 2
Ks = [np.array([[ 1285.6727845238743, 0.0, u0], [ 0.0, 1288.7572378482073, v0], [ 0.0, 0.0, 1.0]]) for _ in range(4)]

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

############ eq(5) Intrinsic residual ############
def residual_intrinsic_eq5(x, pts3D, pts2D, R, t, img_size):
    fx, fy, u0, v0 = x
    RT = np.hstack((R,t))
    ptsW_h = np.hstack([pts3D, np.ones((len(pts3D),1))])
    P = np.vstack((RT,[0,0,0,1]))
    ptsC = (P@ptsW_h.T).T
    Xc=ptsC[:,0]; Yc=ptsC[:,1]; Zc=ptsC[:,2]
    u=pts2D[:,0]; v=pts2D[:,1]

    diff_u = Zc*u - (fx*Xc + u0*Zc)
    diff_v = Zc*v - (fy*Yc + v0*Zc)
    return np.hstack([diff_u, diff_v])

def optimize_intrinsic(pts3D,pts2D,R,t,K_init,img_size):
    x0=[K_init[0,0],K_init[1,1],K_init[0,2],K_init[1,2]]
    res=least_squares(residual_intrinsic_eq5,x0,args=(pts3D,pts2D,R,t,img_size))
    fx,fy,u0,v0=res.x
    return np.array([[fx,0,u0],[0,fy,v0],[0,0,1]])

############ eq(9) Extrinsic residual ############
def residual_extrinsic_eq9(x, pts3D, pts2D, K, R_fixed, cam_id, h):
    tvec=x
    rvec,_=cv2.Rodrigues(R_fixed)
    pts2D_proj,_=cv2.projectPoints(pts3D,rvec,tvec,K,None)
    pts2D_proj=pts2D_proj.reshape(-1,2)
    residuals = (pts2D_proj - pts2D).ravel()
    if cam_id == 1:
        norm_t = np.linalg.norm(tvec)
        penalty = h*(norm_t-1.0)
        residuals = np.append(residuals, penalty)
    return residuals

def optimize_extrinsic_eq9(pts3D,pts2D,K,R_fixed,t_init,cam_id,h):
    x0=t_init.ravel()
    res=least_squares(residual_extrinsic_eq9,x0,args=(pts3D,pts2D,K,R_fixed,cam_id,h))
    t_opt=res.x.reshape(3,1)
    return t_opt

def compute_error(pts3D,pts2D,R,t,K):
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

def initialize_camera_i(i, cam_name, paired_keypoints_list, pts_cam0, pts_cam1, K0, Kref, Rref, tref, points_3D):
    print(f"\n[INFO] Initializing camera {i} ({cam_name})...")
    pts2D_i = extract_points_for_single_camera(paired_keypoints_list, pts_cam0, pts_cam1, cam_name)
    if len(pts2D_i)<6:
        print(f"[WARN] Not enough points for camera {i} ({cam_name})")
        return Ks[i],np.eye(3),np.zeros((3,1))
    print(f"[DEBUG] Extracted {len(pts2D_i)} points for camera {i}")

    min_len = min(len(points_3D), len(pts2D_i))
    pts3D = points_3D[:min_len]
    pts2D_i = pts2D_i[:min_len]

    retval,rvec_i,tvec_i=cv2.solvePnP(pts3D,pts2D_i,Ks[i],None)
    R_i,_=cv2.Rodrigues(rvec_i)
    t_i=tvec_i
    print(f"[DEBUG] Initial R,t for camera {i}:\nR:\n{R_i}\nt:\n{t_i}")

    # Extrinsic 최적화: cam_id=i, cam0 기준이라 t고정, cam1에 penalty, cam2, cam3는 no penalty
    h=10.0
    if i==0:
        # ref camera no extrinsic update
        pass
    elif i==1:
        t_i = optimize_extrinsic_eq9(pts3D,pts2D_i,Ks[i],R_i,t_i,cam_id=1,h=h)
    else:
        t_i = optimize_extrinsic_eq9(pts3D,pts2D_i,Ks[i],R_i,t_i,cam_id=i,h=0.0)

    Ki=optimize_intrinsic(pts3D,pts2D_i,R_i,t_i,Ks[i],image_size)
    print(f"[INFO] Camera {i} ({cam_name}) initialization done.")
    return Ki,R_i,t_i,pts2D_i,pts3D

#############################################
# 메인 실행부
cam_dirs = [
    r'C:\Users\gns15\OneDrive\Desktop\Calibration_With_Keypoints2\cal_json1',
    r'C:\Users\gns15\OneDrive\Desktop\Calibration_With_Keypoints2\cal_json2',
    r'C:\Users\gns15\OneDrive\Desktop\Calibration_With_Keypoints2\cal_json3',
    r'C:\Users\gns15\OneDrive\Desktop\Calibration_With_Keypoints2\cal_json4'
]
paired_keypoints_list=extract_high_confidence_keypoints(cam_dirs,0.55)

pts_cam0, pts_cam1 = extract_points_for_two_cameras(paired_keypoints_list,"json1","json2")
F,mask=compute_fundamental_matrix(pts_cam0,pts_cam1)
K0=Ks[0]
K1_=Ks[1]

E=compute_essential_matrix(F,K0,K1_)
R,t,_=recover_extrinsic_parameters(E,K0,pts_cam0,pts_cam1)
points_3D=triangulate_points(K0,K1_,R,t,pts_cam0,pts_cam1)

# 초기 Intrinsic 최적화 & Joint optim for cam1만
K1_opt=optimize_intrinsic(points_3D,pts_cam1,R,t,K1_,image_size)

print("\n[INFO] Starting initial joint optimization with eq(5) and eq(9) for cam1...")
h=10.0
for i in range(10):
    pts3D=triangulate_points(K0,K1_opt,R,t,pts_cam0,pts_cam1)
    e0=compute_error(pts3D,pts_cam1,R,t,K1_opt)
    print(f"Iteration {i} - Before extrinsic opt: {e0:.4f}")
    # cam1 extrinsic
    t=optimize_extrinsic_eq9(pts3D,pts_cam1,K1_opt,R,t,cam_id=1,h=h)
    pts3D=triangulate_points(K0,K1_opt,R,t,pts_cam0,pts_cam1)
    e1=compute_error(pts3D,pts_cam1,R,t,K1_opt)
    print(f"Iteration {i} - After extrinsic opt: {e1:.4f}")
    K1_opt=optimize_intrinsic(pts3D,pts_cam1,R,t,K1_opt,image_size)
    pts3D=triangulate_points(K0,K1_opt,R,t,pts_cam0,pts_cam1)
    e2=compute_error(pts3D,pts_cam1,R,t,K1_opt)
    print(f"Iteration {i} - After intrinsic opt: {e2:.4f}\n")

print("[INFO] Initial Joint optimization (cam0&1) completed.")
print("Camera1 final:")
print("K:\n",K1_opt)
print("R:\n",R)
print("t:\n",t)

cam2_name="json3"
cam3_name="json4"
K2_final,R2_final,t2_final,pts2D_cam2,pts3D_cam2=initialize_camera_i(2,cam2_name,paired_keypoints_list,pts_cam0,pts_cam1,K0,K1_opt,R,t,points_3D)
K3_final,R3_final,t3_final,pts2D_cam3,pts3D_cam3=initialize_camera_i(3,cam3_name,paired_keypoints_list,pts_cam0,pts_cam1,K0,K1_opt,R,t,points_3D)

print("Camera2:\nK:\n",K2_final,"\nR:\n",R2_final,"\nt:\n",t2_final)
print("Camera3:\nK:\n",K3_final,"\nR:\n",R3_final,"\nt:\n",t3_final)
print("All cameras initialized according to eq(5) and eq(9).")

##############################
# 이후 전체 카메라 Extrinsic & Intrinsic joint optimization
# 로직: Extrinsic 최적화 시: cam0 기준, cam1~cam3 순서로 t최적화(cam1에 penalty)
# Intrinsic 최적화 시: cam0, cam1, cam2, cam3 순서로 K최적화 (Extrinsic 고정)
# 이를 여러 번 반복

# pts2D_cam0, pts2D_cam1는 이미 있음
# pts3D는 cam0,K1_opt,R,t로 triangulate
# cam2, cam3도 pts2D_cam2, pts2D_cam3 확보됨
# 단, cam2, cam3 최적화 시 pts3D 다시 이용 가능(모든 카메라 공통 3D 재구성 필요)
def extrinsic_optim_all_cameras(Ks, R, t_list, pts2D_list, cam_id_order, K0, R0, t0, h):
    # pts2D_list: {cam_id: (pts2D, pts3D)}
    # cam0 is ref, no optimization
    pts3D_all = triangulate_points(K0, Ks[1], R, t_list[1], pts_cam0, pts_cam1)
    # Extrinsic opt:
    # cam_id=1 with penalty
    t_list[1]=optimize_extrinsic_eq9(pts3D_all,pts2D_list[1],Ks[1],R,t_list[1],1,h)
    # cam2, cam3 no penalty
    for cid in [2,3]:
        t_list[cid]=optimize_extrinsic_eq9(pts3D_all,pts2D_list[cid],Ks[cid],R,t_list[cid],cid,0.0)
    return t_list

def intrinsic_optim_all_cameras(Ks, R, t_list, pts2D_list, K0):
    # pts3D 재구성
    pts3D_all = triangulate_points(K0, Ks[1], R, t_list[1], pts_cam0, pts_cam1)
    # cam0~cam3 intrinsic opt
    # cam0 (ref), cam1, cam2, cam3
    for cid in [0,1,2,3]:
        min_len = min(len(pts3D_all), len(pts2D_list[cid]))
        Ks[cid]=optimize_intrinsic(pts3D_all[:min_len], pts2D_list[cid][:min_len], R, t_list[cid], Ks[cid], image_size)
    return Ks

# 준비: pts2D_list, t_list, R, Ks
t_list = [np.zeros((3,1)), t, t2_final, t3_final]
Ks[1]=K1_opt; Ks[2]=K2_final; Ks[3]=K3_final
R0=np.eye(3); t0=np.zeros((3,1))

# pts2D_list 준비
min_len_c2 = min(len(points_3D), len(pts2D_cam2))
min_len_c3 = min(len(points_3D), len(pts2D_cam3))
pts2D_list = {
    0: pts_cam0[:len(points_3D)],
    1: pts_cam1[:len(points_3D)],
    2: pts2D_cam2[:min_len_c2],
    3: pts2D_cam3[:min_len_c3]
}

h=10.0
print("\n[INFO] Full multi-camera optimisation with eq(5) and eq(9):")
for i in range(5):
    # Extrinsic opt all cams
    t_list = extrinsic_optim_all_cameras(Ks, R, t_list, pts2D_list, [0,1,2,3], K0, R0, t0, h)
    # Intrinsic opt all cams
    Ks = intrinsic_optim_all_cameras(Ks, R, t_list, pts2D_list, K0)
    pts3D_all = triangulate_points(K0, Ks[1], R, t_list[1], pts_cam0, pts_cam1)
    err_all=compute_error(pts3D_all,pts_cam1,R,t_list[1],Ks[1])
    print(f"Iteration {i} - avg error with all cams: {err_all:.4f}")

print("All cameras fully optimized with eq(5) and eq(9).")

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
