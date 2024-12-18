from re import T
import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import random
import os
import json
import pprint
import toml
from scipy.optimize import least_squares
from keypoints_confidence_multi_1207 import extract_high_confidence_keypoints
from write_to_toml_v2 import write_calib_to_toml
from visualization import plot_all_camera_poses
import matplotlib.animation as animation

################################
# User-defined constants
################################
image_size = (3840.0, 2160.0)
u0 = image_size[0] / 2
v0 = image_size[1] / 2
# 기준 카메라(cam0): R=I, t=0 고정
ref_R = np.eye(3)
ref_t = np.zeros((3,1))

################################
# Provided camera intrinsics (initial)
################################
K1 = np.array([
    [ 1824.6097978600892, 0.0, 1919.5],
    [ 0.0, 1826.6675222017589, 1079.5],
    [ 0.0, 0.0, 1.0]
])
f_init = K1[0,0]
# 초기에는 f만 쓸 것이므로 u0,v0고정, f_init 사용
def K_from_f(f):
    return np.array([[f, 0, u0],
                     [0, f, v0],
                     [0, 0, 1]])

K2 = K_from_f(f_init)
K3 = K_from_f(f_init)
K4 = K_from_f(f_init)
Ks = [K_from_f(f_init), K_from_f(f_init), K_from_f(f_init), K_from_f(f_init)]


################################
# Functions (no unnecessary changes except where needed)
################################

def cam_create_projection_matrix(K, R, t):
    RT = np.hstack([R, t.reshape(-1,1)])
    return K @ RT

def compute_fundamental_matrix(paired_points):
    points1, points2 = paired_points
    points1 = np.array(points1, dtype=float).reshape(-1, 2)
    points2 = np.array(points2, dtype=float).reshape(-1, 2)
    F, _ = cv2.findFundamentalMat(points1, points2, cv2.FM_RANSAC)
    return F

def compute_essential_matrix(F,K1,K2):
    return K2.T @ F @ K1

def recover_pose_from_essential_matrix(E,pts1,pts2,K):
    pts1 = pts1.astype(np.float32)
    pts2 = pts2.astype(np.float32)
    _, R,t,mask = cv2.recoverPose(E,pts1,pts2,K)
    return R,t,mask

def triangulate_points(paired_keypoints_list, P1, P2):
    points_3d = []
    for (p1, p2) in paired_keypoints_list:
        (x1,y1)=p1; (x2,y2)=p2
        point_3d_h = cv2.triangulatePoints(P1,P2,
                                           np.array([[x1],[y1]],dtype=np.float64),
                                           np.array([[x2],[y2]],dtype=np.float64))
        point_3d = (point_3d_h[:3]/point_3d_h[3]).flatten()
        points_3d.append(point_3d)
    return points_3d

def compute_reprojection_error(points_3d, keypoints_detected, P):
    total_error=0
    total_points=0
    for point_3d,(_,pt2) in zip(points_3d,keypoints_detected):
        Xh = np.append(point_3d,1)
        proj2 = P@Xh
        proj2/=proj2[2]
        error2 = np.linalg.norm(proj2[:2]-np.array(pt2))
        total_error+=error2
        total_points+=1
    return total_error/total_points if total_points>0 else 0

def create_paired_inlier(inliers1,inliers2):
    return [((p1[0],p1[1]),(p2[0],p2[1])) for p1,p2 in zip(inliers1,inliers2)]

def unpack_keypoints(paired_keypoints_list,ref_cam_name,target_cam_name):
    points1, points2=[],[]
    if not isinstance(paired_keypoints_list,list):
        return points1, points2
    total_valid=0
    for frame_data in paired_keypoints_list:
        keypoints_dict = frame_data.get('keypoints', {})
        for kp_idx, cams_dict in keypoints_dict.items():
            if ref_cam_name in cams_dict and target_cam_name in cams_dict:
                p1 = cams_dict[ref_cam_name]
                p2 = cams_dict[target_cam_name]
                if len(p1)==2 and len(p2)==2:
                    points1.append(p1)
                    points2.append(p2)
                    total_valid+=1
    return points1, points2


###########################
# eq(5) for Intrinsic:
# we optimize only f
# residual_intrinsic_eq5: return (Zc*u-(fXc+u0Zc)) and (Zc*v-(fYc+v0Zc))
def residual_intrinsic_eq5(f, pts3D, keypoints2d, R, t, img_size):
    f_val=f[0]
    # Build transformation
    RT=np.hstack((R,t))
    P=np.vstack((RT,[0,0,0,1]))
    residuals=[]
    for p3d,(u,v) in zip(pts3D,keypoints2d):
        Xw,Yw,Zw=p3d
        Xh=np.array([Xw,Yw,Zw,1])
        Pc=P@Xh
        Xc,Yc,Zc=Pc[:3]
        # eq(5)
        diff_u=Zc*u-(f_val*Xc+u0*Zc)
        diff_v=Zc*v-(f_val*Yc+v0*Zc)
        residuals.extend([diff_u,diff_v])
    return np.array(residuals)

def optimize_intrinsic_parameters(points_3d,points2d,R,t):
    # optimize only f
    x0=[f_init]
    res=least_squares(residual_intrinsic_eq5,x0,args=(points_3d, [pt[1] for pt in points2d],R,t,image_size),
                      method='trf', ftol=1e-12, xtol=1e-12, gtol=1e-12, loss='huber')
    f_opt=res.x[0]
    K_opt=K_from_f(f_opt)
    return K_opt

###########################
# eq(9) for extrinsic:
# optimize t only, R fixed, cam_id=2일때 penalty: h*(|t|-1)
def residual_extrinsic_eq9(tvec, pts3D, pts2d, K, R, cam_id, h, multi):
    tvec=tvec.reshape(3,1)
    # reprojection
    P=cam_create_projection_matrix(K,R,tvec)
    residuals=[]
    for p3d,(p1,p2) in zip(pts3D,pts2d):
        Xh=np.append(p3d,1)
        proj=P@Xh
        proj/=proj[2]
        u_est,v_est=proj[0],proj[1]
        u_obs,v_obs=p2
        residuals.extend([u_est-u_obs,v_est-v_obs])
    # eq(9)
    if cam_id==2 and multi==True:
        # print(f"apply penalty: {h}")
        norm_t=np.linalg.norm(tvec)
        penalty=h*(norm_t-1.0)
        residuals.append(penalty)
    return np.array(residuals)

def optimize_extrinsic_parameters(points_3d,points2d,K,R,t,cam_id=1,h=10.0,multi=False):
    # t만 최적화
    x0=t.flatten()
    res=least_squares(residual_extrinsic_eq9,x0,args=(points_3d, points2d, K, R, cam_id, h, multi),
                      method='trf', ftol=1e-12, xtol=1e-12, gtol=1e-12, loss='huber')
    t_opt=res.x.reshape(3,1)
    return t_opt


###########################
# Main code
###########################

ref_cam_dir = r'C:\Users\5W555A\Desktop\Calibration_with_keypoints\demo_lod\json1' # reference camera dir
other_cam_dirs = [r'C:\Users\5W555A\Desktop\Calibration_with_keypoints\demo_lod\json2', r'C:\Users\5W555A\Desktop\Calibration_with_keypoints\demo_lod\json3', r'C:\Users\5W555A\Desktop\Calibration_with_keypoints\demo_lod\json4']

# ref_cam_dir = r'C:\Users\5W555A\Desktop\Calibration_with_keypoints\cal_json1' # reference camera dir
# other_cam_dirs = [r'C:\Users\5W555A\Desktop\Calibration_with_keypoints\cal_json2', r'C:\Users\5W555A\Desktop\Calibration_with_keypoints\cal_json3', r'C:\Users\5W555A\Desktop\Calibration_with_keypoints\cal_json4']

cam_dirs=[ref_cam_dir]+other_cam_dirs
confidence_threshold=0.6
paired_keypoints_list=extract_high_confidence_keypoints(cam_dirs,confidence_threshold)
print(f"paired_keypoints_list: {paired_keypoints_list[:1]}")

# Animate keypoints across all frames
def animate_keypoints(paired_keypoints_list, cam_dirs):
    if not paired_keypoints_list:
        print("No keypoints found!")
        return
    
    n_cameras = len(cam_dirs)
    
    # Create figure and axes
    fig, axes = plt.subplots(1, n_cameras, figsize=(5*n_cameras, 5))
    if n_cameras == 1:
        axes = [axes]
    
    # Initialize scatter plots
    scatters = []
    for idx, cam_dir in enumerate(cam_dirs):
        cam_name = os.path.basename(cam_dir)
        ax = axes[idx]
        scatter = ax.scatter([], [], c='red', marker='o')
        scatters.append(scatter)
        
        ax.set_title(f'Camera: {cam_name}')
        ax.grid(True)
        ax.invert_yaxis()
    
    # Find global min/max coordinates for consistent axis limits
    x_min, x_max = float('inf'), float('-inf')
    y_min, y_max = float('inf'), float('-inf')
    
    for frame in paired_keypoints_list:
        keypoints = frame['keypoints']
        for kp_data in keypoints.values():
            for cam_name in kp_data:
                x, y = kp_data[cam_name]
                x_min = min(x_min, x)
                x_max = max(x_max, x)
                y_min = min(y_min, y)
                y_max = max(y_max, y)
    
    padding = 100
    for ax in axes:
        ax.set_xlim(x_min - padding, x_max + padding)
        ax.set_ylim(y_max + padding, y_min - padding)
    
    # Animation update function
    def update(frame_idx):
        frame = paired_keypoints_list[frame_idx]
        frame_num = frame['frame']
        
        for cam_idx, cam_dir in enumerate(cam_dirs):
            cam_name = os.path.basename(cam_dir)
            x_coords = []
            y_coords = []
            
            keypoints = frame['keypoints']
            for kp_idx, kp_data in keypoints.items():
                if cam_name in kp_data:
                    x, y = kp_data[cam_name]
                    x_coords.append(x)
                    y_coords.append(y)
            
            scatters[cam_idx].set_offsets(np.c_[x_coords, y_coords])
            axes[cam_idx].set_title(f'Camera: {cam_name}\nFrame #{frame_num}')
        
        return scatters
    
    # Create animation
    anim = animation.FuncAnimation(
        fig, update, frames=len(paired_keypoints_list),
        interval=1, blit=False, repeat=True
    )
    
    plt.tight_layout()
    plt.show()

# Call the animation function
animate_keypoints(paired_keypoints_list, cam_dirs)


Fix_K1=K_from_f(f_init) # ref cam K
P1=cam_create_projection_matrix(Fix_K1, ref_R, ref_t) # cam0 fixed

optimization_results={}
all_best_results={}
inlier_pairs_list=[]
fundamental_matrices={}

iteration_binocular = 10
is_multi_camera = False

print("Starting binocular stereo calibration...")
for j,K_init in enumerate(Ks):
    if j==0:
        continue
    ref_cam='json1'
    target_cam=f'json{j+1}'
    points1,points2=unpack_keypoints(paired_keypoints_list,ref_cam,target_cam)
    if len(points1)==0:
        continue
    F=compute_fundamental_matrix((points1,points2))
    fundamental_matrices[(1,j+1)]=F

    # recover pose from E
    E=compute_essential_matrix(F,Fix_K1,K_init) 
    # cam0 is ref:R=I,t=0 fixed, so we only interpret R,t as cam1 pose relative to cam0
    pts1_=np.array(points1)
    pts2_=np.array(points2)
    R_rel,t_rel,_=recover_pose_from_essential_matrix(E,pts1_,pts2_,Fix_K1)

    # create paired inlier
    inlier_pair=create_paired_inlier(pts1_,pts2_)
    inlier_pairs_list.append(inlier_pair)

    # initial
    # cam_id=j+1
    cam_id=j+1
    K2_ = K_from_f(f_init) # start from f_init
    P2=cam_create_projection_matrix(K2_,R_rel,t_rel)
    points_3d=triangulate_points(inlier_pair,P1,P2)
    initial_err=compute_reprojection_error(points_3d,inlier_pair,P2)
    print(f"Initial reprojection error cam pair (1,{cam_id}): {initial_err}")

    for i in range(iteration_binocular):
        # optimize intrinsic (f only)
        K2_=optimize_intrinsic_parameters(points_3d,inlier_pair,R_rel,t_rel)
        P2=cam_create_projection_matrix(K2_,R_rel,t_rel)
        points_3d=triangulate_points(inlier_pair,P1,P2)
        err_int=compute_reprojection_error(points_3d,inlier_pair,P2)
        print(f"After intrinsic opt cam pair (1,{cam_id}): {err_int}")

        # optimize extrinsic (t only, eq(9) if cam_id=1, here cam_id>1 means no penalty)
        # cam_id=1이면 penalty, 아니면 h=0.0 (하지만, binocular 에서는 panalty 없음)
        h = 0.0
        t_opt=optimize_extrinsic_parameters(points_3d, inlier_pair, K2_, R_rel, t_rel, cam_id, h, multi=is_multi_camera)
        P2=cam_create_projection_matrix(K2_,R_rel,t_opt)
        points_3d=triangulate_points(inlier_pair,P1,P2)
        err_ext=compute_reprojection_error(points_3d,inlier_pair,P2)
        print(f"After extrinsic opt cam pair (1,{cam_id}): {err_ext}")

    all_best_results[(1,cam_id)] = {
        'K1': Fix_K1,
        'K2': K2_,
        'R': R_rel,
        't': t_opt,
        'error': err_ext
    }

write_calib_to_toml(all_best_results,image_size=[3840.0,2160.0],output_file="binocular_calib.toml")

# Store binocular calibration results, extracting just the camera index
binocular_poses = {key[1]: (result['R'], result['t']) for key, result in all_best_results.items()}

print("Starting multi-camera calibration...")
reference_points_3d=triangulate_points(inlier_pairs_list[0],P1,
cam_create_projection_matrix(all_best_results[(1,2)]['K2'],all_best_results[(1,2)]['R'],all_best_results[(1,2)]['t']))
print(f"Number of reference 3D points: {len(reference_points_3d)}")

iteration=5
is_multi_camera = True

# Store all camera poses history
all_camera_poses_history = {}

for cam_idx in range(2,len(Ks)+1):
    if (1,cam_idx) not in all_best_results:
        continue
    current_pair=all_best_results[(1,cam_idx)]
    current_K=current_pair['K2']
    current_R=current_pair['R']
    # R fixed
    current_t=current_pair['t']
    current_keypoints=inlier_pairs_list[cam_idx-2]
    current_P=cam_create_projection_matrix(current_K,current_R,current_t)
    init_err=compute_reprojection_error(reference_points_3d,current_keypoints,current_P)
    print(f"Initial reprojection error for camera {cam_idx}: {init_err}")

    # Store camera poses history for this camera
    camera_poses_history = [(current_R.copy(), current_t.copy())]

    # Extrinsic opt (t only) with eq(9) if cam_id=1 else h=0
    h=10.0 if cam_idx==2 else 0.0
    for i in range(iteration):
        # extrinsic opt
        current_t=optimize_extrinsic_parameters(reference_points_3d, current_keypoints, current_K, current_R, current_t, cam_idx,h, multi=is_multi_camera)
        current_P=cam_create_projection_matrix(current_K,current_R,current_t)
        final_err=compute_reprojection_error(reference_points_3d,current_keypoints,current_P)
        print(f"Iteration {i}, camera {cam_idx}, extrinsic opt error: {final_err}")

        # Intrinsic opt f only
        current_K=optimize_intrinsic_parameters(reference_points_3d,current_keypoints,current_R,current_t)
        current_P=cam_create_projection_matrix(current_K,current_R,current_t)
        final_err=compute_reprojection_error(reference_points_3d,current_keypoints,current_P)
        print(f"Iteration {i}, camera {cam_idx}, intrinsic opt error: {final_err}")
        
        # Store camera pose after each iteration
        camera_poses_history.append((current_R.copy(), current_t.copy()))

    # Store this camera's pose history
    all_camera_poses_history[cam_idx] = camera_poses_history

    all_best_results[(1,cam_idx)] = {
        'K1': Fix_K1,
        'K2': current_K,
        'R': current_R,
        't': current_t,
        'error': final_err
    }

# Plot all camera poses after optimization is complete
plot_all_camera_poses(all_camera_poses_history, binocular_poses)

write_calib_to_toml(all_best_results,image_size=[3840.0,2160.0],output_file="multi_calib.toml")
print("Multi-camera calibration completed with eq(5), eq(9), f-only, rotation fixed, cam0 ref fixed.")
