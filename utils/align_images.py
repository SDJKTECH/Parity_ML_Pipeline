import cv2
import numpy as np

"""
Feature-based image alignment using ORB and RANSAC Homography.
Returns the rectified image, the 3x3 homography matrix H, and the valid field-of-view mask.
"""


def align_images(
    master_img: np.ndarray,
    daily_img: np.ndarray,
    max_feature: int = 5000,
    match_ratio: float = 0.75,
    ransac_thres: float = 4.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Aligns daily_img to the perspective of master_img.

  Args:
      master_img: Golden baseline reference image (BGR).
      daily_img: Current image captured by the worker (BGR).
      max_feature: Maximum number of ORB keypoints to retain.
      match_ratio: Lowe's ratio test threshold.
      ransac_thres: Maximum reprojection error allowed for RANSAC inliers.

  Returns:
      tuple:
          - aligned_img (np.ndarray): Perspective-warped image matching
          master_img dimensions.
          - matrix (np.ndarray): 3x3 Homography transformation matrix (maps
          daily -> master).
          - valid_mask (np.ndarray): Binary mask (255 where warped pixels exist,
          0 for black out-of-bounds border).
  """
  if master_img is None or daily_img is None:
    raise ValueError("One or both images passed to align_images are None.")

  master_gray = cv2.cvtColor(master_img, cv2.COLOR_BGR2GRAY)
  daily_gray = cv2.cvtColor(daily_img, cv2.COLOR_BGR2GRAY)

  # 1. Initialize ORB feature detector
  orb = cv2.ORB_create(max_feature)
  keypoints_daily, descriptors_daily = orb.detectAndCompute(daily_gray, None)
  keypoints_master, descriptors_master = orb.detectAndCompute(master_gray, None)

  if descriptors_daily is None or descriptors_master is None:
    raise ValueError(
        "Failed to compute descriptors. Image may be blank, overexposed, or"
        " severely blurred."
    )

  # 2. Match features using Brute-Force Matcher with Hamming Distance
  bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
  matches = bf.knnMatch(descriptors_daily, descriptors_master, k=2)

  # 3. Apply Lowe's ratio test to filter ambiguous matches
  good_matches = []
  for match_pair in matches:
    if len(match_pair) == 2:
      m, n = match_pair
      if m.distance < match_ratio * n.distance:
        good_matches.append(m)

  print(f"[ALIGN] Daily keypoints : {len(keypoints_daily)}")
  print(f"[ALIGN] Master keypoints: {len(keypoints_master)}")
  print(f"[ALIGN] Valid matches   : {len(good_matches)}")

  if len(good_matches) < 8:
    raise ValueError(
        f"Alignment failed: Only {len(good_matches)} valid features found."
        " Prompt worker to retake photo closer to baseline angle."
    )

  # 4. Extract matching point coordinates
  points_daily = np.float32(
      [keypoints_daily[m.queryIdx].pt for m in good_matches]
  ).reshape(-1, 1, 2)
  points_master = np.float32(
      [keypoints_master[m.trainIdx].pt for m in good_matches]
  ).reshape(-1, 1, 2)

  # 5. Estimate 3x3 Homography Matrix (mapping daily_img -> master_img)
  matrix, inliers = cv2.findHomography(
      points_daily, points_master, cv2.RANSAC, ransac_thres
  )

  if matrix is None:
    raise ValueError(
        "Homography calculation failed. The perspective difference is too"
        " extreme."
    )

  height, width = master_img.shape[:2]

  # 6. Warp the worker's image onto the master reference plane
  aligned_img = cv2.warpPerspective(daily_img, matrix, (width, height))

  # 7. Create a binary visibility mask (255 = camera captured, 0 = black unwarped wedge)
  ones = np.full(daily_img.shape[:2], 255, dtype=np.uint8)
  valid_mask = cv2.warpPerspective(
      ones, matrix, (width, height), flags=cv2.INTER_NEAREST
  )

  return aligned_img, matrix, valid_mask