from tomlkit import document, table, array
import cv2
import numpy as np

def nparray_to_list(nparray):
    return [list(row) for row in nparray]

def write_calib_to_toml(all_best_results, image_size=[3840.0, 2160.0], output_file="calib.toml"):
    """
    Write calibration results to a TOML file.
    
    Args:
        all_best_results (dict): Dictionary containing calibration results for each camera pair
                                Format: {(1, cam_idx): {'K1': K1, 'K2': K2, 'R': R, 't': t, 'error': error}}
        image_size (list): Image dimensions [height, width]
        output_file (str): Output TOML filename
    """
    if not all_best_results:
        print("No calibration results to save")
        return
        
    # Initialize the TOML document
    doc = document()
    
    # Get reference camera K (should be the same for all pairs)
    ref_K = None
    for pair_key in all_best_results:
        if 'K1' in all_best_results[pair_key]:
            ref_K = all_best_results[pair_key]['K1']
            break
    
    if ref_K is None:
        print("No reference camera intrinsics found")
        return
    
    # Add reference camera (camera 1)
    camera_data = table()
    camera_data.add("name", "int_cam1_img")
    camera_data.add("size", array(list(image_size)))
    camera_data.add("matrix", array(nparray_to_list(ref_K)))
    camera_data.add("distortions", array([0.0, 0.0, 0.0, 0.0]))
    camera_data.add("rotation", array([0.0, 0.0, 0.0]))
    camera_data.add("translation", array([0.0, 0.0, 0.0]))
    camera_data.add("fisheye", False)
    doc.add("int_cam1_img", camera_data)
    
    # Process each camera pair
    errors = {}
    for pair_key, results in all_best_results.items():
        cam_idx = pair_key[1]  # This is already 1-based index
        
        camera_data = table()
        camera_data.add("name", f"int_cam{cam_idx}_img")
        camera_data.add("size", array(list(image_size)))
        camera_data.add("matrix", array(nparray_to_list(results['K2'])))
        camera_data.add("distortions", array([0.0, 0.0, 0.0, 0.0]))
        
        # Convert rotation matrix to Rodrigues vector
        rvec, _ = cv2.Rodrigues(results['R'])
        camera_data.add("rotation", array(list(rvec.squeeze())))
        
        # Normalize translation vector
        t_normalized = results['t'] / np.linalg.norm(results['t'])
        camera_data.add("translation", array(list(t_normalized.squeeze())))
        
        camera_data.add("fisheye", False)
        doc.add(f"int_cam{cam_idx}_img", camera_data)
        
        errors[cam_idx-1] = results['error']  # Convert to 0-based index for errors

    # Add metadata
    metadata = table()
    metadata.add("adjusted", True)
    metadata.add("avg_error", float(np.mean([err for err in errors.values() if err is not None])))
    metadata.add("errors", {f"cam{k+1}": float(v) if v is not None else 0.0 for k,v in errors.items()})
    doc.add("metadata", metadata)

    # Write toml to file
    with open(output_file, "w") as toml_file:
        toml_file.write(doc.as_string())
    
    print(f"Calibration results saved to {output_file}")
