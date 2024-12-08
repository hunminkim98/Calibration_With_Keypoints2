from tomlkit import document, table, array
import cv2

def nparray_to_list(nparray):
    return [list(row) for row in nparray]

def write_to_toml(all_best_results, set_size = [ 2136.0, 1424.0]):
    # Initialize the TOML document
    doc = document()

    # Create camera data for reference camera
    do_once = False
    # Create camera data for the rest of the other cameras
    for pair_key, results in all_best_results.items():
        if not do_once: # Add reference camera once
            camera_data = table()
            camera_data.add("name", f"int_cam{pair_key[6]}_img")
            camera_data.add("size", array(set_size))
            camera_data.add("matrix", array(nparray_to_list(all_best_results[pair_key]['K1'])))
            camera_data.add("distortions", array([0.0, 0.0, 0.0, 0.0]))
            camera_data.add("rotation", array([0.0, 0.0, 0.0]))
            camera_data.add("translation", array([0.0, 0.0, 1.0]))
            camera_data.add("fisheye", False)
            # Adds the camera data to the TOML document
            doc.add(f"int_cam{pair_key[6]}_img", camera_data)
            do_once = True
        
        # Convert rotation matrix to Rodrigues vector
        rvec, _ = cv2.Rodrigues(results['R'])
        # 3840.0, 2160.0
        camera_data = table()
        camera_data.add("name", f"int_cam{pair_key[-1]}_img") # ex: string pair_key = Camera0_1 where pair_key[-1] is the last character '1'
        camera_data.add("size", array(set_size))
        camera_data.add("matrix", array(nparray_to_list(results['K2'])))
        camera_data.add("distortions", array([0.0, 0.0, 0.0, 0.0]))
        camera_data.add("rotation", array(list(rvec.squeeze())))
        camera_data.add("translation", array(list(results['t'])))
        camera_data.add("fisheye", False)
        

        doc.add(f"int_cam{pair_key[-1]}_img", camera_data)

    # Add metadata
    metadata = table()
    metadata.add("adjusted", False)
    metadata.add("error", 0.0)
    doc.add("metadata", metadata)

    # Write toml to file
    with open("output.toml", "w") as toml_file:
        toml_file.write(doc.as_string())
