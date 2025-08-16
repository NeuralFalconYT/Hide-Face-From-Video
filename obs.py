from video_process import add_mask
mask_names=["Front Man Mask", "Guards Mask", "Red Mask", "Blue Mask"]
camera_index = 0
add_mask(camera_index,mask_name=mask_names[0],mask_up=10, mask_down=10,display=True)

# mask_up mean how much to move the mask up (forehead side)
# mask_down mean how much to move the mask down (chin side)

# Then use OBS to capture the hidden face window. You can use it 
# for live streaming or recording video without showing your face.
