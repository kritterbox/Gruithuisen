import numpy as np
import cv2
import matplotlib.pyplot as plt

def detect_and_normalize_circle_ellipse(image_path):
    """
    Detect the solar disk using circle detection first, then refine with ellipse fitting
    and normalize to a perfect circle. Best of both worlds approach.
    Returns the normalized circular disk, its center, and effective radius.
    """
    # Loading image in gray levels
    orig_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if orig_image is None:
        return None, None, None
        
    height, width = orig_image.shape
    print(f"Image dimensions: {width} x {height} pixels")
   
    # Calculate the scaling factor
    max_dimension = 1000
    scale_factor = min(max_dimension / width, max_dimension / height)
    
    # Calculate new dimensions
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    
    # Resize the image
    image = cv2.resize(orig_image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original image")
    plt.imshow(orig_image, cmap='gray')
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.title("Resized image")
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.show()
    
    # Step 1: Circle Detection (for initial localization)
    blurred = cv2.GaussianBlur(image, (11, 11), 0)
    min_radius = int(300)
    max_radius = int(600)
    
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=100,
                               param1=50, param2=30, 
                               minRadius=min_radius, maxRadius=max_radius)
    
    if circles is None:
        print("No circle detected")
        return None, None, None
    
    # Get the best circle
    circles = np.round(circles[0, :]).astype("int")
    x_circle, y_circle, r_circle = circles[0]
    print(f"Circle detected: Center=({x_circle}, {y_circle}), Radius={r_circle}")
    
    # Step 2: Check if ellipse refinement is needed by measuring the actual solar disk boundary
    ellipse_params = analyze_solar_disk_shape(image, x_circle, y_circle, r_circle)
    
    if ellipse_params is not None:
        center, axes, angle = ellipse_params
        aspect_ratio = max(axes) / min(axes)
        print(f"Ellipse detected: Center=({center[0]:.1f}, {center[1]:.1f}), "
              f"Axes=({axes[0]:.1f}, {axes[1]:.1f}), Angle={angle:.1f}°")
        print(f"Aspect ratio: {aspect_ratio:.3f}")
        
        # Only use ellipse if it's a significant improvement and makes sense
        if aspect_ratio > 1.02 and aspect_ratio < 1.5:  # 2% minimum difference, max 50% elongation
            print(f"Using ellipse refinement (aspect ratio: {aspect_ratio:.3f})")
            use_ellipse = True
            ellipse = ellipse_params
        else:
            print(f"Ellipse not significant enough (aspect ratio: {aspect_ratio:.3f}), using circle")
            use_ellipse = False
            ellipse = ((x_circle, y_circle), (2*r_circle, 2*r_circle), 0)
    else:
        print("No ellipse detected, using original circle")
        use_ellipse = False
        ellipse = ((x_circle, y_circle), (2*r_circle, 2*r_circle), 0)
    
    # Step 3: Extract and normalize the solar disk
    center, axes, angle = ellipse
    cx, cy = int(center[0]), int(center[1])
    
    if use_ellipse:
        # Create elliptical mask
        mask = np.zeros_like(image, dtype=np.uint8)
        ellipse_for_mask = ((int(center[0]), int(center[1])), 
                           (int(axes[0]), int(axes[1])), 
                           int(angle))
        cv2.ellipse(mask, ellipse_for_mask, 255, -1)
        
        # Extract parameters for transformation
        semi_major = int(max(axes) / 2)
        semi_minor = int(min(axes) / 2)
    else:
        # Create circular mask (fallback)
        mask = np.zeros_like(image, dtype=np.uint8)
        cv2.circle(mask, (cx, cy), r_circle, 255, -1)
        
        semi_major = semi_minor = r_circle
        angle = 0
    
    # Extract solar disk with white background
    solar_disk = cv2.bitwise_and(image, mask)
    white_background = np.ones_like(image, dtype=np.uint8) * 255
    inverse_mask = cv2.bitwise_not(mask)
    background = cv2.bitwise_and(white_background, inverse_mask)
    solar_disk_with_bg = cv2.bitwise_or(solar_disk, background)
    
    # Crop to bounding box
    crop_radius = max(semi_major, semi_minor, r_circle) + 20
    x1 = max(cx - crop_radius, 0)
    y1 = max(cy - crop_radius, 0)
    x2 = min(cx + crop_radius, image.shape[1])
    y2 = min(cy + crop_radius, image.shape[0])
    
    cropped = solar_disk_with_bg[y1:y2, x1:x2]
    center_in_crop = (cx - x1, cy - y1)
    
    # Step 4: Transform to perfect circle if ellipse was used
    if use_ellipse and (semi_major != semi_minor or angle != 0):
        print("Applying ellipse-to-circle transformation...")
        final_disk = transform_ellipse_to_circle(cropped, center_in_crop, 
                                               semi_major, semi_minor, angle)
        if final_disk is None:
            print("Transformation failed, using cropped ellipse")
            final_disk = cropped
            final_center = center_in_crop
            effective_radius = max(semi_major, semi_minor)
        else:
            final_center = (final_disk.shape[1]//2, final_disk.shape[0]//2)
            effective_radius = max(semi_major, semi_minor)
    else:
        final_disk = cropped
        final_center = center_in_crop
        effective_radius = r_circle
    
    # Display results
    plt.figure(figsize=(20, 5))
    plt.subplot(1, 4, 1)
    plt.title("Image with Circle Detection")
    img_with_circle = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cv2.circle(img_with_circle, (x_circle, y_circle), r_circle, (0, 255, 0), 2)
    plt.imshow(img_with_circle)
    plt.axis('off')
    
    plt.subplot(1, 4, 2)
    plt.title("Image with Ellipse Refinement")
    img_with_ellipse = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    ellipse_int = ((int(ellipse[0][0]), int(ellipse[0][1])), 
                   (int(ellipse[1][0]), int(ellipse[1][1])), 
                   int(ellipse[2]))
    cv2.ellipse(img_with_ellipse, ellipse_int, (255, 0, 0), 2)
    plt.imshow(img_with_ellipse)
    plt.axis('off')
    
    plt.subplot(1, 4, 3)
    plt.title("Cropped Solar Disk")
    plt.imshow(cropped, cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 4, 4)
    plt.title("Normalized Circular Disk")
    plt.imshow(final_disk, cmap='gray')
    plt.axis('off')
    plt.show()
    
    return final_disk, final_center, effective_radius


def analyze_solar_disk_shape(image, cx, cy, radius):
    """
    Analyze the actual solar disk boundary to determine if it's elliptical.
    This replaces the problematic ROI-based ellipse detection.
    """
    print(f"Analyzing solar disk shape around center ({cx}, {cy}) with radius {radius}")
    
    # Sample the boundary at many angles
    num_samples = 72  # Every 5 degrees
    angles = np.linspace(0, 2*np.pi, num_samples, endpoint=False)
    
    boundary_distances = []
    valid_samples = 0
    
    for angle in angles:
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        
        # Find the actual boundary along this ray
        boundary_distance = find_boundary_along_ray(image, cx, cy, cos_a, sin_a, radius)
        
        if boundary_distance is not None:
            boundary_distances.append(boundary_distance)
            valid_samples += 1
        else:
            boundary_distances.append(radius)  # Fallback to circle radius
    
    if valid_samples < num_samples * 0.5:  # Less than 50% valid samples
        print(f"Insufficient valid boundary samples ({valid_samples}/{num_samples})")
        return None
    
    boundary_distances = np.array(boundary_distances)
    
    # Analyze the pattern to detect elliptical shape
    # Fit ellipse to the boundary points
    boundary_points = []
    for i, (angle, dist) in enumerate(zip(angles, boundary_distances)):
        x = cx + dist * np.cos(angle)
        y = cy + dist * np.sin(angle)
        boundary_points.append([x, y])
    
    boundary_points = np.array(boundary_points, dtype=np.float32)
    
    try:
        # Fit ellipse to boundary points
        ellipse = cv2.fitEllipse(boundary_points.reshape(-1, 1, 2))
        center, axes, angle = ellipse
        
        # Validate the fitted ellipse
        if validate_ellipse_fit(boundary_points, ellipse, cx, cy, radius):
            return ellipse
        else:
            print("Fitted ellipse failed validation")
            return None
            
    except cv2.error as e:
        print(f"Ellipse fitting failed: {e}")
        return None


def find_boundary_along_ray(image, cx, cy, cos_a, sin_a, expected_radius):
    """
    Find the actual solar disk boundary along a ray from the center.
    """
    # Search range
    start_r = max(expected_radius * 0.7, 50)
    end_r = min(expected_radius * 1.3, min(image.shape) // 2 - 10)
    
    if start_r >= end_r:
        return None
    
    # Sample points along the ray
    test_radii = np.linspace(start_r, end_r, 50)
    intensities = []
    
    for r in test_radii:
        x = cx + r * cos_a
        y = cy + r * sin_a
        
        # Check bounds
        if 3 <= x <= image.shape[1] - 4 and 3 <= y <= image.shape[0] - 4:
            # Sample a small area for stability
            x_int, y_int = int(x), int(y)
            intensity = np.mean(image[y_int-1:y_int+2, x_int-1:x_int+2])
            intensities.append(intensity)
        else:
            return None
    
    if len(intensities) < 10:
        return None
    
    # Smooth and find gradient
    from scipy.ndimage import gaussian_filter1d
    smoothed = gaussian_filter1d(intensities, sigma=1.0)
    gradients = np.gradient(smoothed)
    
    # Find the strongest positive gradient (disk edge - dark to light)
    positive_gradients = np.where(gradients > 0)[0]
    
    if len(positive_gradients) > 0:
        max_grad_idx = positive_gradients[np.argmax(gradients[positive_gradients])]
        if max_grad_idx < len(test_radii):
            return test_radii[max_grad_idx]
    
    # Fallback: find median crossing
    median_intensity = np.median(intensities)
    for i, intensity in enumerate(intensities):
        if intensity > median_intensity:
            return test_radii[i]
    
    return None


def validate_ellipse_fit(boundary_points, ellipse, original_cx, original_cy, original_radius):
    """
    Validate that the fitted ellipse makes sense for a solar disk.
    """
    center, axes, angle = ellipse
    
    # Check 1: Center shouldn't drift too much
    center_drift = np.sqrt((center[0] - original_cx)**2 + (center[1] - original_cy)**2)
    max_drift = original_radius * 0.2  # Maximum 20% of radius
    
    if center_drift > max_drift:
        print(f"Ellipse center drift too large: {center_drift:.1f} > {max_drift:.1f}")
        return False
    
    # Check 2: Size should be reasonable
    avg_axis = np.mean(axes)
    expected_diameter = 2 * original_radius
    size_ratio = avg_axis / expected_diameter
    
    if size_ratio < 0.7 or size_ratio > 1.3:
        print(f"Ellipse size unreasonable: {size_ratio:.3f} (should be 0.7-1.3)")
        return False
    
    # Check 3: Aspect ratio should be reasonable for solar disks
    aspect_ratio = max(axes) / min(axes)
    
    if aspect_ratio > 1.5:  # Too elongated for a solar disk
        print(f"Ellipse too elongated: aspect ratio {aspect_ratio:.3f} > 1.5")
        return False
    
    # Check 4: How well does the ellipse fit the boundary points?
    fit_quality = calculate_ellipse_fit_quality(boundary_points, ellipse)
    
    if fit_quality < 0.8:  # Require good fit
        print(f"Ellipse fit quality too low: {fit_quality:.3f} < 0.8")
        return False
    
    print(f"Ellipse validation passed: drift={center_drift:.1f}, size_ratio={size_ratio:.3f}, "
          f"aspect_ratio={aspect_ratio:.3f}, fit_quality={fit_quality:.3f}")
    
    return True


def calculate_ellipse_fit_quality(points, ellipse):
    """
    Calculate how well the ellipse fits the boundary points.
    Returns a score between 0 and 1 (1 = perfect fit).
    """
    center, axes, angle = ellipse
    cx, cy = center
    a, b = axes[0] / 2, axes[1] / 2  # Semi-axes
    
    if a == 0 or b == 0:
        return 0
    
    # Convert angle to radians
    angle_rad = np.radians(angle)
    cos_t, sin_t = np.cos(angle_rad), np.sin(angle_rad)
    
    # Calculate distance from each point to the ellipse
    distances = []
    
    for point in points:
        x, y = point
        
        # Translate to ellipse center
        x_rel = x - cx
        y_rel = y - cy
        
        # Rotate to align with ellipse axes
        x_rot = x_rel * cos_t + y_rel * sin_t
        y_rot = -x_rel * sin_t + y_rel * cos_t
        
        # Calculate normalized distance from ellipse
        normalized_dist = (x_rot / a) ** 2 + (y_rot / b) ** 2
        
        # Distance from ellipse (should be ~1.0 for points on ellipse)
        distance_error = abs(normalized_dist - 1.0)
        distances.append(distance_error)
    
    # Calculate fit quality
    mean_error = np.mean(distances)
    std_error = np.std(distances)
    
    # Good fit: low mean error and low variance
    quality = 1.0 / (1.0 + mean_error * 5 + std_error * 3)
    
    return quality


def transform_ellipse_to_circle(image, center, semi_major, semi_minor, angle):
    """
    Transform an elliptical region to a circular one.
    """
    if semi_major == 0 or semi_minor == 0:
        return None
        
    cx, cy = center
    
    # Calculate scaling factors
    if semi_major >= semi_minor:
        scale_x = 1.0
        scale_y = semi_major / semi_minor
    else:
        scale_x = semi_minor / semi_major
        scale_y = 1.0
    
    # Target size
    target_radius = max(semi_major, semi_minor)
    target_size = int(2 * target_radius + 40)
    
    # Create transformation matrix
    angle_rad = np.radians(-angle)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    
    # Combined transformation matrix
    M = np.array([
        [scale_x * cos_a, -scale_x * sin_a, target_size//2 - scale_x * (cx * cos_a - cy * sin_a)],
        [scale_y * sin_a, scale_y * cos_a, target_size//2 - scale_y * (cx * sin_a + cy * cos_a)]
    ], dtype=np.float32)
    
    try:
        # Apply transformation
        circular_image = cv2.warpAffine(image, M, (target_size, target_size), 
                                       flags=cv2.INTER_LINEAR,
                                       borderMode=cv2.BORDER_CONSTANT,
                                       borderValue=255)
        return circular_image
    except Exception as e:
        print(f"Transformation error: {e}")
        return None