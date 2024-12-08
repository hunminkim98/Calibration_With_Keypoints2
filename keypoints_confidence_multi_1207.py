import os
import json

def extract_high_confidence_keypoints(cam_dirs, confidence_threshold):
    """
    Extracts high-confidence keypoints (x, y) from all cameras simultaneously,
    only if all cameras meet the confidence threshold for the corresponding keypoint.
    
    Args:
        - cam_dirs: List of directories containing JSON files for each camera.
        - confidence_threshold: Confidence value threshold for keypoints.
        
    Returns:
        - high_confidence_keypoints: A list of high-confidence keypoints for each frame,
                                     where each element is a dictionary with camera names as keys
                                     and corresponding keypoints as values.
    """
    high_confidence_keypoints = []
    
    # 각 카메라별 키포인트 카운터 초기화
    camera_keypoint_counts = {}
    
    # Get sorted list of JSON files for each camera
    cam_files = {}
    for cam_dir in cam_dirs:
        cam_name = os.path.basename(cam_dir)
        cam_files[cam_name] = sorted([os.path.join(cam_dir, f) for f in os.listdir(cam_dir) if f.endswith('.json')])
        camera_keypoint_counts[cam_name] = 0  # 카메라별 카운터 초기화
    
    # Iterate over frames simultaneously for all cameras
    for frame_files in zip(*cam_files.values()):
        frame_keypoints = {}
        
        # Extract keypoints and confidence for each camera
        cam_keypoints = {}
        for cam_name, frame_file in zip(cam_files.keys(), frame_files):
            with open(frame_file, 'r') as file:
                data = json.load(file)
                
                if data['people']:
                    keypoints = data['people'][0]['pose_keypoints_2d']
                    keypoints_conf = [(keypoints[i], keypoints[i+1], keypoints[i+2]) for i in range(0, len(keypoints), 3)]
                    cam_keypoints[cam_name] = keypoints_conf
        
        # Check if all cameras have the same number of keypoints
        if len(set(len(kp) for kp in cam_keypoints.values())) == 1:
            # Filter keypoints based on confidence threshold across all cameras
            for i in range(len(next(iter(cam_keypoints.values())))):
                if all(cam_keypoints[cam][i][2] >= confidence_threshold for cam in cam_keypoints):
                    kp_coords = {cam: (cam_keypoints[cam][i][0], cam_keypoints[cam][i][1]) for cam in cam_keypoints}
                    frame_keypoints[i] = kp_coords
                    # 각 카메라의 키포인트 카운트 증가
                    for cam in cam_keypoints:
                        camera_keypoint_counts[cam] += 1
        
        if frame_keypoints:
            high_confidence_keypoints.append(frame_keypoints)
    
    # 각 카메라별 추출된 키포인트 수 출력
    print("\nNumber of extracted keypoints per camera:")
    for cam_name, count in camera_keypoint_counts.items():
        print(f"{cam_name}: {count} keypoints")
    
    return high_confidence_keypoints