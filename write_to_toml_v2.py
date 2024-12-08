from tomlkit import document, table, array
import cv2
import numpy as np

def nparray_to_list(nparray):
    return [list(row) for row in nparray]

def write_calib_to_toml(Ks, R, t_list, errors, image_size=[1088.0, 1920.0], output_file="Calib.toml"):
    # Initialize the TOML document
    doc = document()
    
    # Add camera data for all cameras
    for cam_idx in range(len(Ks)):
        camera_data = table()
        camera_data.add("name", f"int_cam{cam_idx+1}_img")
        camera_data.add("size", array(list(image_size)))
        camera_data.add("matrix", array(nparray_to_list(Ks[cam_idx])))
        camera_data.add("distortions", array([0.0, 0.0, 0.0, 0.0]))
        
        if cam_idx == 0:  # Reference camera
            camera_data.add("rotation", array([0.0, 0.0, 0.0]))
            camera_data.add("translation", array([0.0, 0.0, 1.0]))
        else:
            # Convert rotation matrix to Rodrigues vector
            rvec, _ = cv2.Rodrigues(R)
            camera_data.add("rotation", array(list(rvec.squeeze())))
            # Normalize translation to ensure last element is 1.0
            t_normalized = t_list[cam_idx] / np.linalg.norm(t_list[cam_idx])
            camera_data.add("translation", array(list(t_normalized.squeeze())))
        
        camera_data.add("fisheye", False)
        doc.add(f"int_cam{cam_idx+1}_img", camera_data)

    # Add metadata
    metadata = table()
    metadata.add("adjusted", True)
    metadata.add("avg_error", float(np.mean([err for err in errors.values() if err is not None])))
    metadata.add("errors", {f"cam{k+1}": float(v) if v is not None else 0.0 for k,v in errors.items()})
    doc.add("metadata", metadata)

    # Write toml to file
    with open(output_file, "w") as toml_file:
        toml_file.write(doc.as_string())
