import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
import random
import os
import json
import pprint
import toml
from keypoints_confidence_multi_1207 import extract_high_confidence_keypoints
from write_to_toml_v2 import write_calib_to_toml
from scipy.optimize import least_squares



# Constants for initial intrinsic matrix ( Factory setting in the paper but im using calibrate app in Matlab or OpenCV )
## It would be changed input data from Pose2Sim intrinsic calibration
image_size = (3840.0, 2160.0)
u0 = image_size[0] / 2
v0 = image_size[1] / 2
K1 = np.array([
    [ 1824.6097978600892, 0.0, 1919.5],
    [ 0.0, 1826.6675222017589, 1079.5],
    [ 0.0, 0.0, 1.0]
])

K2 = np.array([
    [ 1824.6097978600892, 0.0, 1919.5],
    [ 0.0, 1826.6675222017589, 1079.5],
    [ 0.0, 0.0, 1.0]
])
K3 = np.array([
    [ 1824.6097978600892, 0.0, 1919.5],
    [ 0.0, 1826.6675222017589, 1079.5],
    [ 0.0, 0.0, 1.0]
])

K4 = np.array([
    [ 1824.6097978600892, 0.0, 1919.5],
    [ 0.0, 1826.6675222017589, 1079.5],
    [ 0.0, 0.0, 1.0]
])

Ks = [K1, K2, K3, K4]
###################### Data Processing ############################

# camera directories
ref_cam_dir = r'C:\Users\5W555A\Desktop\Calibration_With_Keypoints2\cal_json1' # reference camera directory
other_cam_dirs = [r'C:\Users\5W555A\Desktop\Calibration_With_Keypoints2\cal_json2', r'C:\Users\5W555A\Desktop\Calibration_With_Keypoints2\cal_json3', r'C:\Users\5W555A\Desktop\Calibration_With_Keypoints2\cal_json4'] # other camera directories
cam_dirs = [ref_cam_dir] + other_cam_dirs
print(f"cam_dirs: {cam_dirs}")
confidence_threshold = 0.85 # confidence threshold for keypoints pair extraction

# Call the function to extract paired keypoints
paired_keypoints_list = extract_high_confidence_keypoints(cam_dirs, confidence_threshold) # checking completed

def unpack_keypoints(paired_keypoints_list, ref_cam_name, target_cam_name):
    """
    Unpacks the paired keypoints for a specific camera pair from all frames.

    Args:
        paired_keypoints_list (list): List of dictionaries containing frame and keypoints data.
        ref_cam_name (str): Name of the reference camera (e.g., 'cal_json1')
        target_cam_name (str): Name of the target camera (e.g., 'cal_json2')

    Returns:
        tuple: A tuple containing two lists, where each list contains the x and y coordinates 
               of the keypoints from reference and target cameras respectively.
    """
    points1, points2 = [], []
    
    if not isinstance(paired_keypoints_list, list):
        print(f"Warning: Input is not a list")
        return points1, points2
    
    print(f"Processing camera pair: {ref_cam_name} (reference) - {target_cam_name} (target)")
    
    # Process all frames
    total_valid_pairs = 0
    for frame_data in paired_keypoints_list:
        frame_idx = frame_data.get('frame', -1)
        keypoints_dict = frame_data.get('keypoints', {})
        frame_valid_pairs = 0
        
        # Process each keypoint in the frame
        for keypoint_idx, keypoint_data in keypoints_dict.items():
            if ref_cam_name in keypoint_data and target_cam_name in keypoint_data:
                ref_point = list(keypoint_data[ref_cam_name])
                target_point = list(keypoint_data[target_cam_name])
                # Basic validation of point coordinates
                if all(isinstance(coord, (int, float)) for coord in ref_point + target_point):
                    points1.append(ref_point)
                    points2.append(target_point)
                    frame_valid_pairs += 1
        
        total_valid_pairs += frame_valid_pairs
        
    print(f"Total valid keypoint pairs extracted: {total_valid_pairs}")
    if total_valid_pairs > 0:
        print(f"Sample of first few points:")
        for i in range(min(3, len(points1))):
            print(f"Pair {i}: Ref: {points1[i]}, Target: {points2[i]}")
    
    return points1, points2


###################### Data Processing ############################

###################### Function of Extrinsics parameters optimisation ############################

def compute_fundamental_matrix(paired_keypoints_list):
    """
    Compute the fundamental matrix from paired keypoints.

    This function takes a list of paired keypoints and computes the fundamental matrix.

    Args:
        paired_keypoints_list (list): A list of tuples, where each tuple contains two arrays of keypoints, one for each image.

    Returns:
        numpy.ndarray: The computed fundamental matrix.
    """
    points1, points2 = paired_keypoints_list
    
    points1 = np.array(points1, dtype=float).reshape(-1, 2)
    points2 = np.array(points2, dtype=float).reshape(-1, 2)

    # Compute the fundamental matrix using RANSAC
    F, _ = cv2.findFundamentalMat(points1, points2, cv2.FM_RANSAC)

    return F


def compute_essential_matrix(F, K1, K2):
    """
    Compute the essential matrix given the fundamental matrix and camera calibration matrices.

    Args:
        F (numpy.ndarray): The fundamental matrix.
        K1 (numpy.ndarray): The calibration matrix of camera 1.
        K2 (numpy.ndarray): The calibration matrix of other camera.

    Returns:
        numpy.ndarray: The computed essential matrix.
    """
    E = K2.T @ F @ K1
    #print(f"Essential matrix: {E}")
    return E

def recover_pose_from_essential_matrix(E, points1_inliers, points2_inliers, K):
    """
    Recover the camera pose from the Essential matrix using inliers.

    Parameters:
    E (numpy.ndarray): The Essential matrix.
    points1_inliers (numpy.ndarray): The inlier points from the first image.
    points2_inliers (numpy.ndarray): The inlier points from the second image.
    K (numpy.ndarray): The camera intrinsic matrix (assuming the same for both cameras).

    Returns:
    numpy.ndarray, numpy.ndarray: The rotation matrix (R) and the translation vector (t).
    """
    # Ensure points are in the correct shape and type
    points1_inliers = points1_inliers.astype(np.float32)
    points2_inliers = points2_inliers.astype(np.float32)

    # Recovering the pose
    _, R, t, mask = cv2.recoverPose(E, points1_inliers, points2_inliers, K)

    return R, t, mask


def cam_create_projection_matrix(K, R, t):
    """
    Creates the camera projection matrix.

    Args:
        K (numpy.ndarray): The camera's intrinsic parameters matrix.
        R (numpy.ndarray): The rotation matrix.
        t (numpy.ndarray): The translation vector.

    Returns:
        numpy.ndarray: The created projection matrix.
    """
    RT = np.hstack([R, t.reshape(-1, 1)])
    return K @ RT


def triangulate_points(paired_keypoints_list, P1, P2):
    """
    Triangulates a list of paired keypoints using the given camera projection matrices.

    Args:
        paired_keypoints_list (list): List of paired keypoints, where each item is a tuple containing 
                                      two sets of coordinates for the same keypoint observed in both cameras.
        P1 (array-like): Camera projection matrix for the reference camera.
        P2 (array-like): Camera projection matrix for the other camera.

    Returns:
        list: List of 3D points corresponding to the triangulated keypoints.
    """
    points_3d = []

    for keypoint_pair in paired_keypoints_list:
        (x1, y1), (x2, y2) = keypoint_pair

        # Convert coordinates to homogeneous format for triangulation
        point_3d_homogeneous = cv2.triangulatePoints(P1, P2, np.array([[x1], [y1]], dtype=np.float64), np.array([[x2], [y2]], dtype=np.float64))

        # Normalize to convert to non-homogeneous 3D coordinates
        point_3d = point_3d_homogeneous[:3] / point_3d_homogeneous[3]

        points_3d.append(point_3d)

    return points_3d

# Visualize the 3D points
def plot_3d_points(points_3d):
    """
    Plots a set of 3D points.

    Args:
        points_3d (list): List of frames, where each frame is a list of 3D points represented as (x, y, z) coordinates.

    Returns:
        None
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for frame in points_3d:
        for point in frame:
            ax.scatter(point[0], point[1], point[2], c='b', marker='o')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()

def compute_reprojection_error(precomputed_points_3d, keypoints_detected, P2):
    """
    Computes the reprojection error for the second camera (P2) using the given projection matrices
    and precomputed 3D points.

    Args:
        precomputed_points_3d (list): List of precomputed 3D points as NumPy arrays.
        keypoints_detected (list): List of paired keypoints, each represented as a tuple (2D point in camera 1, 2D point in camera 2).
        P2 (array-like): Camera projection matrix for the other camera.

    Returns:
        float: The mean reprojection error for the second camera.
    """
    total_error = 0
    total_points = 0

    # Ensure the length of 3D points matches the 2D keypoints
    assert len(precomputed_points_3d) == len(keypoints_detected), "Number of 3D points and 2D keypoints must match"

    # Process each pair of 3D point and 2D keypoints
    for point_3d, (_, point2) in zip(precomputed_points_3d, keypoints_detected):
        # Convert 3D point to homogeneous coordinates
        point_3d_homogeneous = np.append(point_3d.flatten(), 1)

        # Reproject the 3D point to the 2D image plane for camera 2
        point2_reprojected = P2 @ point_3d_homogeneous
        point2_reprojected /= point2_reprojected[2]

        # Compute reprojection error for camera 2's reprojected point
        error2 = np.linalg.norm(point2_reprojected[:2] - np.array(point2))

        total_error += error2
        total_points += 1

    mean_error = total_error / total_points if total_points > 0 else 0
    return mean_error

###################### Function of Intrinsics parameters optimisation ############################

def compute_intrinsic_optimization_loss(x, points_3d, keypoints_detected, R, t):
    """
    Computes the loss for intrinsic parameters optimization.

    Args:
        - x: Intrinsic parameters to optimize (f_x, f_y, u0, v0).
        - points_3d: List of 3D points as arrays.
        - keypoints_detected: 2D inlier points, each row is a pair (u, v).
        - R: Rotation matrix.
        - t: Translation vector.

    Returns:
        - The mean loss for the intrinsic parameters optimization.
    """
    f_x, f_y, u0, v0 = x  # Intrinsic parameters to optimize
    dx = 1.0  # Pixel scaling factor dx
    dy = 1.0  # Pixel scaling factor dy

    total_loss = 0
    valid_keypoints_count = 0

    # Build the homogeneous transformation matrix
    transformation_matrix = np.hstack((R, t.reshape(-1, 1)))
    transformation_matrix = np.vstack((transformation_matrix, [0, 0, 0, 1]))

    # Make sure the number of 3D points matches the 2D keypoints
    assert len(points_3d) == len(keypoints_detected), "Number of 3D points and 2D keypoints must match"

    # Process each point
    for point_3d, detected_point in zip(points_3d, keypoints_detected):
        if not isinstance(detected_point, (list, tuple, np.ndarray)) or len(detected_point) != 2:
            continue

        u_detected, v_detected = detected_point
        valid_keypoints_count += 1

        # Convert 3D point to homogeneous coordinates
        point_3d_homogeneous = np.append(point_3d.flatten(), 1)
        point_camera = transformation_matrix.dot(point_3d_homogeneous)
        Xc, Yc, Zc = point_camera[:3]

        # Compute the loss based on the difference between expected and detected points
        loss = abs(Zc * u_detected - ((f_x / dx) * Xc + u0 * Zc)) + abs(Zc * v_detected - ((f_y / dy) * Yc + v0 * Zc))
        total_loss += loss

    mean_loss = total_loss / valid_keypoints_count if valid_keypoints_count > 0 else 0
    print(f"mear_loss of intrinsic : {mean_loss}")
    return mean_loss

def optimize_intrinsic_parameters(points_3d, keypoints_detected, K, R, t):
    """
    Optimizes the intrinsic parameters using the given 3D points and detected keypoints.

    Args:
    - points_3d: List of 3D points (triangulated human body joints).
    - keypoints_detected: Original detected 2D keypoints.
    - K: Intrinsic parameters matrix.
    - R: Rotation matrix.
    - t: Translation vector.

    Returns:
    - The optimized intrinsic parameters matrix.
    """
    # Create the initial guess for the intrinsic parameters
    x0 = np.array([K[0, 0], K[1,1] ,K[0, 2], K[1, 2]])
     
    # Create the bounds for the intrinsic parameters
    bounds = ([0, 0, 0, 0], [np.inf, np.inf, np.inf, np.inf])

    # Optimize the intrinsic parameters using the least squares method
    result = least_squares(compute_intrinsic_optimization_loss, x0, args=(points_3d, keypoints_detected, R, t), bounds=bounds, x_scale='jac', verbose=1, method='trf', loss= 'huber', diff_step=1e-8, tr_solver='lsmr', ftol=1e-12, max_nfev=50, xtol=1e-12, gtol=1e-12)

    # Create the optimized intrinsic matrix
    K_optimized = np.array([[result.x[0], 0, result.x[2]], [0, result.x[1], result.x[3]], [0, 0, 1]])

    return K_optimized

def create_paired_inlier(inliers1, inliers2):
    """
    Creates a list of paired inliers.

    Args:
        inliers1 (numpy.ndarray): Array of inlier points from camera 1.
        inliers2 (numpy.ndarray): Array of inlier points from camera 2.

    Returns:
        list of tuples: Each tuple contains paired points (tuples), 
                        where each sub-tuple is a point (x, y) from camera 1 and camera 2 respectively.
    """
    paired_inliers = [((p1[0], p1[1]), (p2[0], p2[1])) for p1, p2 in zip(inliers1, inliers2)]
    return paired_inliers

###################### Function of Intrinsics parameters optimisation ############################

###################### Function of Extrinsic parameters optimisation ############################
def compute_extrinsic_optimization_loss(x, ext_K, points_3d, points_2d, ext_R, P2):
    """
    Computes the loss for the extrinsic parameters optimization using reprojection error.

    Args:
    - x: Extrinsic parameters to optimize (translation vector).
    - ext_K: Intrinsic parameters matrix.
    - points_3d: List of 3D points (triangulated human body joints).
    - points_2d: Original detected 2D keypoints.
    - ext_R: Rotation matrix.

    Returns:
    - The mean reprojection error.
    """
    
    # Calculate reprojection error using the existing function
    mean_error = compute_reprojection_error(points_3d, points_2d, P2)
    print(f"mean_error of extrinsic : {mean_error}")

    return mean_error

def optimize_extrinsic_parameters(points_3d, other_cameras_keypoints, ext_K, ext_R, ext_t, P2):
    """
    Optimizes the extrinsic parameters using the given 3D points and detected keypoints.

    Args:
    - points_3d: List of 3D points (triangulated human body joints).
    - other_cameras_keypoints: Original detected 2D keypoints for the other cameras.
    - ext_K: Intrinsic parameters matrix.
    - ext_R: Rotation matrix.
    - ext_t: Translation vector.

    Returns:
    - The optimized t vector.
    """
    # Create the initial guess for the extrinsic parameters (|T|) using the t vector magnitude
    x0 = ext_t.flatten()

    # Optimize the intrinsic parameters using the least squares method
    result = least_squares(compute_extrinsic_optimization_loss, x0, args=(ext_K, points_3d, other_cameras_keypoints, ext_R, P2), verbose=1, method='trf', diff_step=1e-8 , ftol=1e-12, max_nfev=150, xtol=1e-12, gtol=1e-12, x_scale='jac', loss='huber')

    optimized_t = result.x # optimized t vector

    return optimized_t

###################### Function of Extrinsic parameters optimisation ############################


###################### Initialization ############################
print("Starting binocular stereo calibration...")
# Initialize variables
num_iterations = 3  # Single iteration count replacing inner and outer iterations
optimization_results = {}
all_best_results = {}

# Preset lists and dictionary for data storage
camera_Rt = {}
inlier_pairs_list = []
inlier2_list = []
fundamental_matrices = {}

# Fix the intrinsic matrix for the reference camera
Fix_K1 = K1
P1 = cam_create_projection_matrix(Fix_K1, np.eye(3), np.zeros(3))
# Iterate over camera pairs, skipping the reference camera
for j, K in enumerate(Ks):
    if j == 0:
        continue  # Skip the reference camera
    
    OPT_K = K

    ref_cam = 'cal_json1'  # 참조 카메라 (camera 1)
    target_cam = f'cal_json{j+1}'  # 대상 카메라 (camera 2, 3, 4, ...)
    
    print(f"Camera {j + 1} relative to Camera 1:")
    points1, points2 = unpack_keypoints(paired_keypoints_list, ref_cam, target_cam)
    
    if len(points1) == 0 or len(points2) == 0:
        print(f"Warning: No valid keypoint pairs found for cameras {ref_cam} and {target_cam}")
        continue

    F = compute_fundamental_matrix((points1, points2))
    points1 = np.array(points1, dtype=float).reshape(-1, 2)
    points2 = np.array(points2, dtype=float).reshape(-1, 2)
    inlier2_list.append(points2)

    inlier_pair = create_paired_inlier(points1, points2)
    inlier_pairs_list.append(inlier_pair)

    fundamental_matrices[(1, j + 1)] = F
    camera_pair_key = (1, j + 1)
    
    paired_keypoints = inlier_pairs_list[j - 1]
    F = fundamental_matrices[(1, j + 1)]

    E = compute_essential_matrix(F, Fix_K1, K)
    R, t, mask = recover_pose_from_essential_matrix(E, points1, points2, Fix_K1)
    print(f"Camera {j + 1} relative to Camera 1: R = {R}, t = {t}")

    camera_Rt[j + 1] = (R, t)
    R_fixed = R
    t_optimized = t

    optimization_results.setdefault(camera_pair_key, {
        'K1': [], 'K2': [], 'R': [], 't': [], 'errors': [], 'losses': []
    })

    # initial triangulation
    P2 = cam_create_projection_matrix(OPT_K, R_fixed, t_optimized)
    points_3d = triangulate_points(paired_keypoints, P1, P2)

    # initial reprojection error
    reprojection_error = compute_reprojection_error(points_3d, paired_keypoints, P2)
    print(f"Initial reprojection error: {reprojection_error}")

    # Single iteration loop for optimization
    for iter_idx in range(num_iterations):    
        # optimize intrinsic parameters, but extrinsic parameters are fixed
        OPT_K = optimize_intrinsic_parameters(points_3d, points2, OPT_K, R_fixed, t_optimized)
        P2 = cam_create_projection_matrix(OPT_K, R_fixed, t_optimized)
        points_3d = triangulate_points(paired_keypoints, P1, P2)
        reprojection_error = compute_reprojection_error(points_3d, paired_keypoints, P2)
        print(f"Camera pair {camera_pair_key} iteration {iter_idx + 1}: intrinsic reprojection error: {reprojection_error}")

        # optimize extrinsic parameters, but intrinsic parameters are fixed
        OPT_t = optimize_extrinsic_parameters(points_3d, paired_keypoints, OPT_K, R_fixed, t_optimized, P2)
        P2 = cam_create_projection_matrix(OPT_K, R_fixed, OPT_t)
        points_3d = triangulate_points(paired_keypoints, P1, P2)
        reprojection_error = compute_reprojection_error(points_3d, paired_keypoints, P2)
        print(f"Camera pair {camera_pair_key} iteration {iter_idx + 1}: extrinsic reprojection error: {reprojection_error}")

        # update t
        t_optimized = OPT_t

        # save results
        optimization_results[camera_pair_key]['K1'].append(Fix_K1)
        optimization_results[camera_pair_key]['K2'].append(OPT_K)
        optimization_results[camera_pair_key]['R'].append(R_fixed)
        optimization_results[camera_pair_key]['t'].append(OPT_t)
        optimization_results[camera_pair_key]['errors'].append(reprojection_error)
        # optimization_results[camera_pair_key]['losses'].append(loss)

    if optimization_results[camera_pair_key]['errors']:
        min_error_for_pair = min(optimization_results[camera_pair_key]['errors'])
        index_of_min_error = optimization_results[camera_pair_key]['errors'].index(min_error_for_pair)
        best_K1 = optimization_results[camera_pair_key]['K1'][index_of_min_error]
        best_K2 = optimization_results[camera_pair_key]['K2'][index_of_min_error]
        best_R = optimization_results[camera_pair_key]['R'][index_of_min_error]
        best_t = optimization_results[camera_pair_key]['t'][index_of_min_error]

        all_best_results[camera_pair_key] = {
            'K1': best_K1,
            'K2': best_K2,
            'R': best_R,
            't': best_t,
            'error': min_error_for_pair
        }

    # save initial results
    write_calib_to_toml(all_best_results)

####################################################
##########EXTRINSIC PARAMETER OPTIMIZATION##########
####################################################

########################################
####### Multi-camera calibration #######
########################################

# N = 20 # how many times to run the optimization

# for i, K in enumerate(Ks):
#     if i == 0:  # skip the reference camera
#         continue
    
#     # keypoints for optimization
#     other_keypoints_detected = inlier2_list[i-1] # use the keypoints for the other camera
#     paired_keypoints_list_multi = inlier_pairs_list[i-1] 

#     pair_key = (1, i+1) # pair key
#     print(f"calibrating camera {i+1}...")

#     # import the best results for each camera pair
#     ext_K = all_best_results[pair_key]['K2'] 
#     ext_R = all_best_results[pair_key]['R']
#     ext_t = all_best_results[pair_key]['t']
#     ref_t = np.zeros(3)  # origin


#     # projection matrix
#     P1 = cam_create_projection_matrix(Ks[0], np.eye(3), ref_t)
#     P2 = cam_create_projection_matrix(ext_K, ext_R, ext_t)

#     # triangulate points
#     points_3d = triangulate_points(paired_keypoints_list_multi, P1, P2) # initial 3D points
#     before_optimization_error = compute_reprojection_error(points_3d, paired_keypoints_list_multi, P1, P2)
#     print(f"camera {i+1} before optimization error: {before_optimization_error}")



#     # Entrinsic and intrinsic parameter joint optimization
#     for n in range(N):

#         # extrinsic parameter optimization
#         print(f"before optimization t vector: {ext_t}")
#         optimized_t = optimize_extrinsic_parameters(points_3d, other_keypoints_detected, ext_K, ext_R, ext_t) # optimize extrinsic parameters
#         ext_t = optimized_t # update t vector
#         print(f"{n + 1}th optimized t vector: {ext_t}")

#         N_P2 = cam_create_projection_matrix(ext_K, ext_R, ext_t) # update projection matrix
#         ex_reprojection_error = compute_reprojection_error(points_3d, paired_keypoints_list_multi, P1, N_P2) # calculate the mean reprojection error
#         print(f"{n + 1}th error in extrinsic optimization = {ex_reprojection_error}")

#         # intrinsic parameter optimization
#         points_3d = triangulate_points(paired_keypoints_list_multi, P1, N_P2) # update 3D points after extrinsic optimization
#         ext_K_optimized = optimize_intrinsic_parameters(points_3d, other_keypoints_detected, ext_K, ext_R, ext_t) # optimize intrinsic parameters
#         ext_K = ext_K_optimized # update intrinsic parameters
#         # print(f"{n + 1}th optimized K matrix: {ext_K}")

#         N_P2 = cam_create_projection_matrix(ext_K, ext_R, ext_t) # update projection matrix
#         in_reprojection_error = compute_reprojection_error(points_3d, paired_keypoints_list_multi, P1, N_P2) # calculate the mean reprojection error
#         print(f"{n + 1}th error in intrinsic optimization = {in_reprojection_error}")
#         points_3d = triangulate_points(paired_keypoints_list_multi, P1, N_P2) # update 3D points after intrinsic optimization

#     # save result after optimization
#     all_best_results[pair_key]['t'] = ext_t
#     all_best_results[pair_key]['K2'] = ext_K

#     # ext_R matrix to rod vector
#     ext_R_rod, _ = cv2.Rodrigues(ext_R)
#     all_best_results[pair_key]['R'] = ext_R_rod

# # print optimized results
# for pair_key, results in all_best_results.items():
#     print(f"Best results for {pair_key}:")
#     print(f"- K2: {results['K2']}") # optimized intrinsic paramters
#     print(f" rod R: {results['R']}") # optimized extrinsic paramters
#     print(f"- t: {results['t']}") # optimized extrinsic paramters


#################################################### intrinsic jointly optimization ####################################################

####################################################
##########EXTRINSIC PARAMETER OPTIMIZATION##########
####################################################