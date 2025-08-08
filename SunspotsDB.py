import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt

def create_sunspot_database(solar_disk, center, radius, intensity_threshold=None, 
                          small_spot_sensitivity="normal", large_spot_sensitivity="normal", 
                          output_csv="sunspot_database.csv"):
    """
    Analyze pixels within a circle and create a database of pixels with intensity below a threshold.
    Uses multi-pass approach to catch both small dark spots and larger fainter spots.
    
    Args:
    - solar_disk: Grayscale image of the solar disk.
    - center: Tuple (x, y) representing the center of the circle.
    - radius: Radius of the circle.
    - intensity_threshold: Intensity threshold for detecting dark pixels. If None, computed automatically.
    - small_spot_sensitivity: Control detection of small spots ("high", "normal", "low")
    - large_spot_sensitivity: Control detection of large/faint spots ("high", "normal", "low")  
    - output_csv: Path to save the output CSV file.
    
    Returns:
    - DataFrame containing positions and intensities of dark pixels.
    """
    height, width = solar_disk.shape
    x_center, y_center = center
    
    print(f"Processing solar disk: {width}x{height}, center=({x_center}, {y_center}), radius={radius}")
    print(f"Small spot sensitivity: {small_spot_sensitivity}")
    print(f"Large spot sensitivity: {large_spot_sensitivity}")
    
    # Step 1: Create efficient mask for solar disk (vectorized)
    y_coords, x_coords = np.ogrid[:height, :width]
    disk_mask = (x_coords - x_center)**2 + (y_coords - y_center)**2 <= radius**2
    
    # Create edge exclusion mask
    edge_buffer = max(3, int(radius * 0.02))
    inner_mask = (x_coords - x_center)**2 + (y_coords - y_center)**2 <= (radius - edge_buffer)**2
    
    # Get intensities of all pixels within the solar disk (excluding edge)
    disk_intensities = solar_disk[inner_mask]
    
    print(f"Solar disk contains {len(disk_intensities)} pixels (excluding {edge_buffer}px edge buffer)")
    print(f"Intensity range: {disk_intensities.min()} - {disk_intensities.max()}")
    print(f"Mean intensity: {disk_intensities.mean():.1f}")
    print(f"Standard deviation: {disk_intensities.std():.1f}")
    
    # Step 2: Multi-threshold approach for different sunspot types
    if intensity_threshold is None:
        # Conservative threshold for small/clear sunspots
        conservative_threshold = compute_conservative_threshold(disk_intensities)
        # More permissive threshold to catch parts of larger sunspots
        permissive_threshold = compute_permissive_threshold(disk_intensities)
        # New: Large spot threshold for fainter but larger features
        large_spot_threshold = compute_large_spot_threshold(disk_intensities, large_spot_sensitivity)
    else:
        conservative_threshold = intensity_threshold
        permissive_threshold = intensity_threshold + 15
        large_spot_threshold = intensity_threshold + 30  # Even more permissive
    
    print(f"Using conservative threshold: {conservative_threshold}")
    print(f"Using permissive threshold: {permissive_threshold}")
    print(f"Using large spot threshold: {large_spot_threshold}")
    
    # Step 3: Find candidates with different thresholds
    conservative_candidates = (solar_disk < conservative_threshold) & inner_mask
    conservative_coords = np.where(conservative_candidates)
    
    permissive_candidates = (solar_disk < permissive_threshold) & inner_mask
    permissive_coords = np.where(permissive_candidates)
    
    large_spot_candidates = (solar_disk < large_spot_threshold) & inner_mask
    large_spot_coords = np.where(large_spot_candidates)
    
    print(f"Conservative candidates: {len(conservative_coords[0])}")
    print(f"Permissive candidates: {len(permissive_coords[0])}")
    print(f"Large spot candidates: {len(large_spot_coords[0])}")
    
    # Step 4: Region-based analysis to group pixels into sunspots
    sunspot_regions = find_sunspot_regions_multi_threshold(
        solar_disk, 
        conservative_coords, permissive_coords, large_spot_coords,
        conservative_threshold, permissive_threshold, large_spot_threshold,
        small_spot_sensitivity, large_spot_sensitivity
    )
    
    print(f"Found {len(sunspot_regions)} sunspot regions")
    
    # Step 5: Validate and extract final sunspot pixels
    valid_pixels = validate_sunspot_regions_enhanced(
        solar_disk, sunspot_regions, x_center, y_center, radius, 
        small_spot_sensitivity, large_spot_sensitivity
    )
    
    print(f"After region validation: {len(valid_pixels)} valid sunspot pixels")
    
    # Step 6: Visualize results
    visualize_sunspot_detection_multi_threshold(
        solar_disk, center, radius, 
        conservative_threshold, permissive_threshold, large_spot_threshold,
        conservative_coords, permissive_coords, large_spot_coords,
        sunspot_regions, valid_pixels
    )
    
    # Step 7: Create DataFrame
    if len(valid_pixels) == 0:
        df = pd.DataFrame()
    else:
        pixel_data = []
        for y, x, spot_type in valid_pixels:
            distance_to_center = np.sqrt((x - x_center)**2 + (y - y_center)**2)
            intensity = solar_disk[y, x]
            pixel_data.append({
                "x": int(x), 
                "y": int(y), 
                "intensity": int(intensity),
                "distance_from_center": round(distance_to_center, 1),
                "relative_distance": round(distance_to_center / radius, 3),
                "spot_type": spot_type  # "small", "medium", "large"
            })
        
        df = pd.DataFrame(pixel_data)
    
    # Step 8: Save results
    df.to_csv(output_csv, index=False)
    
    print(f"Base de données sauvegardée dans le fichier : {output_csv}")
    print(f"Nombre total de pixels détectés: {len(df)}")
    
    if len(df) > 0:
        spot_types = df['spot_type'].value_counts()
        print(f"Spot type distribution: {dict(spot_types)}")
    
    return df


def find_sunspot_regions_multi_threshold(image, conservative_coords, permissive_coords, large_spot_coords,
                                        conservative_thresh, permissive_thresh, large_spot_thresh,
                                        small_sensitivity, large_sensitivity):
    """
    Enhanced region finding that handles multiple threshold levels for different spot types.
    FIXED: Better detection and classification of large vs small spots.
    """
    from scipy import ndimage
    
    # Create binary masks for all thresholds
    height, width = image.shape
    conservative_mask = np.zeros((height, width), dtype=bool)
    permissive_mask = np.zeros((height, width), dtype=bool)
    large_spot_mask = np.zeros((height, width), dtype=bool)
    
    if len(conservative_coords[0]) > 0:
        conservative_mask[conservative_coords] = True
    if len(permissive_coords[0]) > 0:
        permissive_mask[permissive_coords] = True
    if len(large_spot_coords[0]) > 0:
        large_spot_mask[large_spot_coords] = True
    
    # Find connected components in each mask
    structure = ndimage.generate_binary_structure(2, 2)
    
    regions = []
    small_params = get_sensitivity_parameters(small_sensitivity)
    large_params = get_large_spot_parameters(large_sensitivity)
    
    # STEP 1: Process conservative regions first (these are definitely small, dark spots)
    labeled_conservative, num_conservative = ndimage.label(conservative_mask, structure)
    print(f"Found {num_conservative} conservative seed regions")
    
    processed_pixels = set()  # Track which pixels we've already assigned to regions
    
    for region_id in range(1, num_conservative + 1):
        region_mask = (labeled_conservative == region_id)
        region_coords = np.where(region_mask)
        region_size = len(region_coords[0])
        
        min_seed_size = max(2, small_params['min_size_central'] // 2)
        if region_size >= min_seed_size:
            # Expand using permissive threshold
            expanded_region = expand_region_multi_threshold(
                image, region_mask, permissive_mask, large_spot_mask, 
                conservative_thresh, permissive_thresh, large_spot_thresh, "small"
            )
            
            if len(expanded_region) >= 3:
                regions.append((expanded_region, "small"))
                # Mark these pixels as processed
                for y, x in expanded_region:
                    processed_pixels.add((y, x))
    
    # STEP 2: Find large regions using a different approach
    # Look for connected components in large_spot_mask that aren't already processed
    remaining_large_mask = large_spot_mask.copy()
    for y, x in processed_pixels:
        if 0 <= y < height and 0 <= x < width:
            remaining_large_mask[y, x] = False
    
    labeled_large, num_large = ndimage.label(remaining_large_mask, structure)
    print(f"Found {num_large} potential large spot regions")
    
    for region_id in range(1, num_large + 1):
        region_mask = (labeled_large == region_id)
        region_coords = np.where(region_mask)
        region_size = len(region_coords[0])
        
        # FIXED: More appropriate size thresholds for large spots
        min_large_seed_size = max(large_params['min_size_central'], 30)  # Ensure minimum reasonable size
        
        if region_size >= min_large_seed_size:
            # For large spots, we want to be more inclusive in the expansion
            expanded_region = expand_region_for_large_spots(
                image, region_mask, large_spot_mask, large_spot_thresh
            )
            
            final_size = len(expanded_region)
            
            # FIXED: Better classification logic
            # Classify as large if it meets size criteria OR has the right intensity characteristics
            if (final_size >= min_large_seed_size or 
                is_likely_large_spot(image, expanded_region, large_spot_thresh)):
                
                regions.append((expanded_region, "large"))
                print(f"Added large spot region with {final_size} pixels")
    
    # STEP 3: Handle remaining permissive regions that might be medium spots
    remaining_permissive_mask = permissive_mask.copy()
    for y, x in processed_pixels:
        if 0 <= y < height and 0 <= x < width:
            remaining_permissive_mask[y, x] = False
    
    # Remove pixels already in large spot regions
    for region_data in regions:
        if len(region_data) == 2 and region_data[1] == "large":
            region = region_data[0]
            for y, x in region:
                remaining_permissive_mask[y, x] = False
    
    labeled_medium, num_medium = ndimage.label(remaining_permissive_mask, structure)
    print(f"Found {num_medium} potential medium spot regions")
    
    for region_id in range(1, num_medium + 1):
        region_mask = (labeled_medium == region_id)
        region_coords = np.where(region_mask)
        region_size = len(region_coords[0])
        
        min_medium_size = small_params['min_size_central']
        if region_size >= min_medium_size:
            expanded_region = list(zip(*region_coords))
            regions.append((expanded_region, "medium"))
    
    print(f"Total regions found: {len(regions)}")
    return regions


def expand_region_for_large_spots(image, seed_region, large_mask, threshold):
    """
    Special expansion logic for large spots that tends to be more inclusive.
    """
    from scipy import ndimage
    
    current_region = seed_region.copy()
    
    # More aggressive expansion for large spots
    for iteration in range(6):  # More iterations
        structure = ndimage.generate_binary_structure(2, 2)  # 8-connectivity
        expanded = ndimage.binary_dilation(current_region, structure, iterations=1)
        
        # Find candidates - pixels that are in the large mask and adjacent to current region
        candidates = expanded & large_mask & ~current_region
        
        if not np.any(candidates):
            break
            
        current_region = current_region | candidates
    
    return list(zip(*np.where(current_region)))


def is_likely_large_spot(image, region, threshold):
    """
    Additional heuristics to identify if a region is likely a large sunspot
    even if it doesn't meet strict size requirements.
    """
    if len(region) < 20:
        return False
    
    # Check intensity characteristics
    y_coords, x_coords = zip(*region)
    intensities = [image[y, x] for y, x in region]
    avg_intensity = np.mean(intensities)
    
    # Large spots tend to have moderate contrast but larger area
    region_array = np.array(region)
    y_min, y_max = region_array[:, 0].min(), region_array[:, 0].max()
    x_min, x_max = region_array[:, 1].min(), region_array[:, 1].max()
    
    bbox_area = (y_max - y_min + 1) * (x_max - x_min + 1)
    fill_ratio = len(region) / bbox_area
    
    # Large spots often have good fill ratio and moderate intensity
    return (avg_intensity < threshold + 20 and fill_ratio > 0.3 and bbox_area > 100)


def validate_sunspot_regions_enhanced(image, regions, x_center, y_center, radius, 
                                    small_sensitivity, large_sensitivity):
    """
    FIXED: Enhanced validation with better handling of large spots.
    """
    valid_pixels = []
    
    small_params = get_sensitivity_parameters(small_sensitivity) 
    large_params = get_large_spot_parameters(large_sensitivity)
    
    for region_data in regions:
        if len(region_data) == 2:
            region, spot_type = region_data
        else:
            region, spot_type = region_data, "small"  # Fallback
        
        # Choose parameters based on spot type
        if spot_type == "large":
            params = large_params
            is_valid = validate_large_spot_region(image, region, x_center, y_center, radius, params)
        else:
            params = small_params
            is_valid = validate_small_medium_spot_region(image, region, x_center, y_center, radius, params, spot_type)
        
        if is_valid:
            # Add spot type to each pixel
            for y, x in region:
                valid_pixels.append((y, x, spot_type))
            print(f"Validated {spot_type} spot with {len(region)} pixels")
        else:
            print(f"Rejected {spot_type} spot with {len(region)} pixels")
    
    return valid_pixels


def validate_large_spot_region(image, region, x_center, y_center, radius, params):
    """
    FIXED: Specialized validation for large spots with more appropriate criteria.
    """
    region_size = len(region)
    
    if region_size == 0:
        return False
    
    # Get region properties
    y_coords, x_coords = zip(*region)
    y_coords, x_coords = np.array(y_coords), np.array(x_coords)
    
    # Calculate distance from center
    region_center_y = np.mean(y_coords)
    region_center_x = np.mean(x_coords)
    distance_from_center = np.sqrt((region_center_x - x_center)**2 + (region_center_y - y_center)**2)
    relative_distance = distance_from_center / radius
    
    # FIXED: More appropriate size requirements for large spots
    if relative_distance < 0.4:
        min_size = max(params['min_size_central'], 25)  # Minimum absolute size
    elif relative_distance < 0.7:
        min_size = max(params['min_size_middle'], 20)
    else:
        min_size = max(params['min_size_outer'], 15)
    
    if region_size < min_size:
        print(f"Large spot rejected: size {region_size} < min_size {min_size}")
        return False
    
    # FIXED: Less restrictive edge exclusion for large spots
    if relative_distance > 0.95:  # Only exclude very edge regions
        print(f"Large spot rejected: too close to edge (distance {relative_distance:.3f})")
        return False
    
    # FIXED: Higher size limit for large spots
    max_size = 8000  # Much larger limit for large spots
    if region_size > max_size:
        print(f"Large spot rejected: too large {region_size} > {max_size}")
        return False
    
    # FIXED: More lenient geometric validation for large spots
    y_min, y_max = np.min(y_coords), np.max(y_coords)
    x_min, x_max = np.min(x_coords), np.max(x_coords)
    
    bbox_area = (y_max - y_min + 1) * (x_max - x_min + 1)
    compactness = region_size / bbox_area
    
    # FIXED: More lenient compactness for large spots (they can be more irregular)
    min_compactness = max(0.1, params['min_compactness_small'] * 0.6)
    if compactness < min_compactness:
        print(f"Large spot rejected: compactness {compactness:.3f} < {min_compactness:.3f}")
        return False
    
    # FIXED: More lenient aspect ratio for large spots
    height_bbox = y_max - y_min + 1
    width_bbox = x_max - x_min + 1
    aspect_ratio = max(height_bbox, width_bbox) / min(height_bbox, width_bbox)
    
    max_aspect_ratio = min(8.0, params['max_aspect_ratio_small'] * 1.5)  # More lenient
    if aspect_ratio > max_aspect_ratio:
        print(f"Large spot rejected: aspect ratio {aspect_ratio:.3f} > {max_aspect_ratio:.3f}")
        return False
    
    # FIXED: Simplified contrast validation for large spots
    region_intensities = [image[y, x] for y, x in region]
    avg_region_intensity = np.mean(region_intensities)
    
    # For large spots, we just need to ensure they're reasonably darker than average
    disk_mean = np.mean(image[image > 0])  # Approximate disk average
    contrast = disk_mean - avg_region_intensity
    
    min_contrast = max(5, params['min_contrast_small'] * 0.4)  # Much more lenient
    if contrast < min_contrast:
        print(f"Large spot rejected: contrast {contrast:.1f} < {min_contrast:.1f}")
        return False
    
    print(f"Large spot validated: size={region_size}, compactness={compactness:.3f}, "
          f"aspect_ratio={aspect_ratio:.3f}, contrast={contrast:.1f}")
    return True


def validate_small_medium_spot_region(image, region, x_center, y_center, radius, params, spot_type):
    """
    Validation for small and medium spots (original logic, mostly unchanged).
    """
    region_size = len(region)
    
    if region_size == 0:
        return False
    
    # Get region properties
    y_coords, x_coords = zip(*region)
    y_coords, x_coords = np.array(y_coords), np.array(x_coords)
    
    # Calculate distance from center
    region_center_y = np.mean(y_coords)
    region_center_x = np.mean(x_coords)
    distance_from_center = np.sqrt((region_center_x - x_center)**2 + (region_center_y - y_center)**2)
    relative_distance = distance_from_center / radius
    
    # Size filtering
    if relative_distance < 0.3:
        min_size = params['min_size_central']
    elif relative_distance < 0.6:
        min_size = params['min_size_middle']
    else:
        min_size = params['min_size_outer']
    
    if region_size < min_size:
        return False
    
    # Edge exclusion
    if relative_distance > params['edge_exclusion_factor']:
        return False
    
    # Size limits
    max_size = 2000
    if region_size > max_size:
        return False
    
    # Geometric validation
    y_min, y_max = np.min(y_coords), np.max(y_coords)
    x_min, x_max = np.min(x_coords), np.max(x_coords)
    
    bbox_area = (y_max - y_min + 1) * (x_max - x_min + 1)
    compactness = region_size / bbox_area
    
    if compactness < params['min_compactness_small']:
        return False
    
    # Aspect ratio
    height_bbox = y_max - y_min + 1
    width_bbox = x_max - x_min + 1
    aspect_ratio = max(height_bbox, width_bbox) / min(height_bbox, width_bbox)
    
    if aspect_ratio > params['max_aspect_ratio_small']:
        return False
    
    # Contrast validation
    region_intensities = [image[y, x] for y, x in region]
    avg_region_intensity = np.mean(region_intensities)
    
    # Sample surrounding area
    surrounding_pixels = []
    sample_size = min(len(region), 20)
    
    for i, (y, x) in enumerate(region[:sample_size]):
        for dy in range(-4, 5):
            for dx in range(-4, 5):
                ny, nx = y + dy, x + dx
                if (0 <= ny < image.shape[0] and 0 <= nx < image.shape[1] and 
                    (ny, nx) not in region):
                    if abs(dy) >= 2 or abs(dx) >= 2:
                        surrounding_pixels.append(image[ny, nx])
    
    if len(surrounding_pixels) > 0:
        avg_surrounding = np.mean(surrounding_pixels)
        contrast = avg_surrounding - avg_region_intensity
        
        min_contrast = params['min_contrast_small']
        
        if contrast < min_contrast:
            return False
    else:
        return False
    
    return True


def visualize_sunspot_detection_multi_threshold(solar_disk, center, radius, 
                                              conservative_thresh, permissive_thresh, large_thresh,
                                              conservative_coords, permissive_coords, large_coords,
                                              regions, valid_pixels):
    """
    FIXED: Enhanced visualization with correct colors and better labeling.
    """
    x_center, y_center = center
    
    plt.figure(figsize=(25, 5))
    
    # Original
    plt.subplot(1, 5, 1)
    annotated_image = cv2.cvtColor(solar_disk, cv2.COLOR_GRAY2BGR)
    cv2.circle(annotated_image, (int(x_center), int(y_center)), int(radius), (0, 255, 0), 2)
    plt.title("Original")
    plt.imshow(annotated_image)
    plt.axis("off")
    
    # Conservative candidates
    plt.subplot(1, 5, 2)
    conservative_image = solar_disk.copy()
    if len(conservative_coords[0]) > 0:
        conservative_image[conservative_coords] = 0
    plt.title(f"Conservative (thresh={conservative_thresh})")
    plt.imshow(conservative_image, cmap='gray')
    plt.axis("off")
    
    # Permissive candidates
    plt.subplot(1, 5, 3)
    permissive_image = solar_disk.copy()
    if len(permissive_coords[0]) > 0:
        permissive_image[permissive_coords] = 0
    plt.title(f"Permissive (thresh={permissive_thresh})")
    plt.imshow(permissive_image, cmap='gray')
    plt.axis("off")
    
    # Large spot candidates
    plt.subplot(1, 5, 4)
    large_image = solar_disk.copy()
    if len(large_coords[0]) > 0:
        large_image[large_coords] = 0
    plt.title(f"Large spots (thresh={large_thresh})")
    plt.imshow(large_image, cmap='gray')
    plt.axis("off")
    
    # FIXED: Final result with correct colors and spot centers
    plt.subplot(1, 5, 5)
    final_image = cv2.cvtColor(solar_disk, cv2.COLOR_GRAY2BGR)
    
    if len(valid_pixels) > 0:
        # Group pixels by spot type for center calculation
        spot_regions_by_type = {"small": [], "medium": [], "large": []}
        
        # Collect regions by type
        for region_data in regions:
            if len(region_data) == 2:
                region, spot_type = region_data
                if spot_type in spot_regions_by_type:
                    spot_regions_by_type[spot_type].append(region)
        
        # FIXED: Draw centers with correct colors
        # Small spots: Green
        for region in spot_regions_by_type["small"]:
            if region:
                y_coords, x_coords = zip(*region)
                center_x, center_y = int(np.mean(x_coords)), int(np.mean(y_coords))
                cv2.circle(final_image, (center_x, center_y), 3, (0, 255, 0), -1)  # Green
        
        # Medium spots: Yellow
        for region in spot_regions_by_type["medium"]:
            if region:
                y_coords, x_coords = zip(*region)
                center_x, center_y = int(np.mean(x_coords)), int(np.mean(y_coords))
                cv2.circle(final_image, (center_x, center_y), 4, (0, 255, 255), -1)  # Yellow
        
        # Large spots: Blue (FIXED: was showing as red before)
        for region in spot_regions_by_type["large"]:
            if region:
                y_coords, x_coords = zip(*region)
                center_x, center_y = int(np.mean(x_coords)), int(np.mean(y_coords))
                cv2.circle(final_image, (center_x, center_y), 6, (255, 0, 0), -1)  # Blue
    
    plt.title(f"Sunspot centers (Green=small, Yellow=medium, Blue=large)")
    plt.imshow(final_image)
    plt.axis("off")
    
    plt.tight_layout()
    plt.show()
    
    # Print summary
    if len(regions) > 0:
        type_counts = {}
        for region_data in regions:
            if len(region_data) == 2:
                _, spot_type = region_data
                type_counts[spot_type] = type_counts.get(spot_type, 0) + 1
        print(f"Final spot type counts: {type_counts}")


def get_large_spot_parameters(sensitivity_level):
    """
    FIXED: Get filtering parameters specifically for large, faint spots with better defaults.
    """
    if sensitivity_level == "high":
        return {
            'min_size_central': 30, 'min_size_middle': 50, 'min_size_outer': 80,
            'min_contrast_small': 5, 'min_contrast_tiny': 8,
            'min_compactness_small': 0.1, 'max_aspect_ratio_small': 8.0, 
            'edge_exclusion_factor': 0.95
        }
    elif sensitivity_level == "low":
        return {
            'min_size_central': 80, 'min_size_middle': 120, 'min_size_outer': 180,
            'min_contrast_small': 15, 'min_contrast_tiny': 18,
            'min_compactness_small': 0.3, 'max_aspect_ratio_small': 4.0,
            'edge_exclusion_factor': 0.8
        }
    else:  # normal
        return {
            'min_size_central': 50, 'min_size_middle': 75, 'min_size_outer': 120,
            'min_contrast_small': 8, 'min_contrast_tiny': 12,
            'min_compactness_small': 0.15, 'max_aspect_ratio_small': 6.0,
            'edge_exclusion_factor': 0.9
        }


# Include all the helper functions from original code
def compute_large_spot_threshold(disk_intensities, sensitivity="normal"):
    """
    Compute a threshold specifically for large, faint sunspots.
    These are often missed by conservative thresholds.
    """
    background_intensity = estimate_solar_background(disk_intensities)
    background_std = estimate_background_std(disk_intensities, background_intensity)
    
    # More permissive options for large spots
    base_thresholds = {
        'bg_minus_1std': background_intensity - 1.0 * background_std,
        'bg_minus_0.5std': background_intensity - 0.5 * background_std,
        'p15': np.percentile(disk_intensities, 15),
        'p20': np.percentile(disk_intensities, 20),
        'p25': np.percentile(disk_intensities, 25),
    }
    
    # Adjust based on sensitivity
    if sensitivity == "high":
        # Even more permissive for large spots
        adjustment_factor = 1.2
        percentile_boost = 5
    elif sensitivity == "low":
        # More conservative
        adjustment_factor = 0.8
        percentile_boost = -5
    else:
        adjustment_factor = 1.0
        percentile_boost = 0
    
    # Apply adjustments
    thresholds = {}
    for method, threshold in base_thresholds.items():
        if 'bg_minus' in method:
            thresholds[method] = threshold * adjustment_factor
        else:  # percentile methods
            percentile_val = float(method[1:]) + percentile_boost
            percentile_val = max(5, min(40, percentile_val))  # Keep in reasonable range
            thresholds[method] = np.percentile(disk_intensities, percentile_val)
    
    # Select threshold that captures 10-40% of pixels (much more permissive)
    valid_thresholds = {}
    for method, threshold in thresholds.items():
        if threshold > 0:
            percentage = (disk_intensities < threshold).sum() / len(disk_intensities) * 100
            if 10.0 <= percentage <= 40.0:
                valid_thresholds[method] = threshold
    
    if valid_thresholds:
        # Use a moderate threshold (not the most permissive to avoid too much noise)
        selected = np.percentile(list(valid_thresholds.values()), 30)  # 30th percentile
        return int(selected)
    else:
        # Fallback: more permissive than conservative
        fallback = background_intensity - 1.0 * background_std
        return max(int(fallback), int(np.percentile(disk_intensities, 20)))


def get_sensitivity_parameters(sensitivity_level):
    """Get filtering parameters based on sensitivity level."""
    if sensitivity_level == "high":
        return {
            'min_size_central': 3, 'min_size_middle': 5, 'min_size_outer': 8,
            'min_size_tiny_validation': 8, 'min_contrast_small': 15, 'min_contrast_tiny': 18,
            'min_compactness_small': 0.3, 'max_aspect_ratio_small': 4.0, 'edge_exclusion_factor': 0.75
        }
    elif sensitivity_level == "low":
        return {
            'min_size_central': 8, 'min_size_middle': 15, 'min_size_outer': 25,
            'min_size_tiny_validation': 20, 'min_contrast_small': 25, 'min_contrast_tiny': 30,
            'min_compactness_small': 0.5, 'max_aspect_ratio_small': 2.5, 'edge_exclusion_factor': 0.9
        }
    else:  # normal
        return {
            'min_size_central': 5, 'min_size_middle': 10, 'min_size_outer': 18,
            'min_size_tiny_validation': 12, 'min_contrast_small': 20, 'min_contrast_tiny': 25,
            'min_compactness_small': 0.4, 'max_aspect_ratio_small': 3.0, 'edge_exclusion_factor': 0.85
        }


def expand_region_multi_threshold(image, seed_region, mask1, mask2, thresh1, thresh2, thresh3, spot_type):
    """Enhanced region expansion that can use multiple thresholds."""
    from scipy import ndimage
    
    current_region = seed_region.copy()
    
    # Choose expansion strategy based on spot type
    if spot_type == "large":
        # For large spots, be more aggressive in expansion
        iterations = 5
        primary_mask = mask1
    else:
        # For small spots, use original strategy
        iterations = 3
        primary_mask = mask1
    
    # Iteratively expand
    for iteration in range(iterations):
        structure = ndimage.generate_binary_structure(2, 1)
        expanded = ndimage.binary_dilation(current_region, structure, iterations=1)
        
        # Find candidates for expansion
        candidates = expanded & primary_mask & ~current_region
        
        if not np.any(candidates):
            # Try with secondary mask if primary expansion failed
            if spot_type == "large" and iteration < 2:
                candidates = expanded & mask2 & ~current_region
        
        if not np.any(candidates):
            break
            
        current_region = current_region | candidates
    
    return list(zip(*np.where(current_region)))


# Helper functions (copied from original working code)
def compute_conservative_threshold(disk_intensities):
    """Compute a conservative threshold that reliably detects clear sunspots."""
    background_intensity = estimate_solar_background(disk_intensities)
    background_std = estimate_background_std(disk_intensities, background_intensity)
    
    thresholds = {
        'bg_minus_2.5std': background_intensity - 2.5 * background_std,
        'bg_minus_3std': background_intensity - 3.0 * background_std,
        'p3': np.percentile(disk_intensities, 3),
        'p5': np.percentile(disk_intensities, 5),
    }
    
    valid_thresholds = {}
    for method, threshold in thresholds.items():
        if threshold > 0:
            percentage = (disk_intensities < threshold).sum() / len(disk_intensities) * 100
            if 1.0 <= percentage <= 8.0:
                valid_thresholds[method] = threshold
    
    if valid_thresholds:
        selected = min(valid_thresholds.values())
        return int(selected)
    else:
        fallback = background_intensity - 3.0 * background_std
        return max(int(fallback), int(np.percentile(disk_intensities, 3)))


def compute_permissive_threshold(disk_intensities):
    """Compute a more permissive threshold to catch parts of larger/fainter sunspots."""
    background_intensity = estimate_solar_background(disk_intensities)
    background_std = estimate_background_std(disk_intensities, background_intensity)
    
    thresholds = {
        'bg_minus_2std': background_intensity - 2.0 * background_std,
        'bg_minus_1.5std': background_intensity - 1.5 * background_std,
        'p8': np.percentile(disk_intensities, 8),
        'p12': np.percentile(disk_intensities, 12),
    }
    
    valid_thresholds = {}
    for method, threshold in thresholds.items():
        if threshold > 0:
            percentage = (disk_intensities < threshold).sum() / len(disk_intensities) * 100
            if 5.0 <= percentage <= 20.0:
                valid_thresholds[method] = threshold
    
    if valid_thresholds:
        selected = np.percentile(list(valid_thresholds.values()), 40)
        return int(selected)
    else:
        conservative = compute_conservative_threshold(disk_intensities)
        return conservative + 15


def estimate_solar_background(intensities):
    """Estimate the typical solar surface background intensity using robust methods."""
    hist, bin_edges = np.histogram(intensities, bins=50)
    mode_bin = np.argmax(hist)
    hist_mode = (bin_edges[mode_bin] + bin_edges[mode_bin + 1]) / 2
    
    trimmed_mean = np.mean(np.percentile(intensities, [10, 90]))
    median_val = np.median(intensities)
    
    candidates = [hist_mode, trimmed_mean, median_val]
    background = np.mean([val for val in candidates if val >= np.percentile(intensities, 40)])
    
    return background


def estimate_background_std(intensities, background_intensity):
    """Estimate the standard deviation of the solar background."""
    background_pixels = intensities[abs(intensities - background_intensity) < 20]
    
    if len(background_pixels) > 50:
        return np.std(background_pixels)
    else:
        return background_intensity * 0.05