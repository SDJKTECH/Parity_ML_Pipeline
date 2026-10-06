import os
import cv2

def identify_room(current_img_path, master_dir="master_images"):
    current_img = cv2.imread(current_img_path, cv2.IMREAD_GRAYSCALE)
    if current_img is None:
        return None
    orb = cv2.ORB_create(nfeatures=500)

    # Extracted keypoints (locations) and 256-bit binary descriptors (signatures)
    kp_curr, des_curr = orb.detectAndCompute(current_img, None)

    if des_curr is None:
        return None

    # Setup for Brute-Force Matcher optimized for binary descriptors (NORM_HAMMING)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    best_match_room = None
    max_good_matches = 0

    valid_exts = (".jpg", ".jpeg", ".png", ".webp")
    for file_name in os.listdir(master_dir):
        if not file_name.lower().endswith(valid_exts):
            continue

        master_path = os.path.join(master_dir, file_name)
        master_img = cv2.imread(master_path, cv2.IMREAD_GRAYSCALE)
        if master_img is None:
            continue

        # Extracted keypoints and descriptors from the candidate master image
        kp_master, des_master = orb.detectAndCompute(master_img, None)
        if des_master is None:
            continue

        # Bitwise compare current descriptors against master descriptors
        matches = bf.match(des_curr, des_master)
        
        # Tracked which room shares the highest number of overlapping features
        if len(matches) > max_good_matches:
            max_good_matches = len(matches)
            best_match_room = os.path.splitext(file_name)[0]

    # Return room name only if it hits a confidence threshold (>15 matched keypoints)
    return best_match_room if max_good_matches > 15 else None