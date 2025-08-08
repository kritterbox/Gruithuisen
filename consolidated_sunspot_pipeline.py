"""
Consolidated Sunspot Detection Pipeline
Streamlined workflow: read -> preprocess -> suggest threshold -> detect -> results
"""

import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import os


def detect_sunspots_complete_pipeline(image_path, output_csv="sunspot_database.csv", 
                                    manual_threshold=None, show_diagnostics=False):
    """
    Complete sunspot detection pipeline with minimal output.
    
    Args:
        image_path: Path to the solar disk image
        output_csv: Path for output CSV file
        manual_threshold: Optional manual threshold (if None, will be suggested)
        show_diagnostics: Whether to show diagnostic plots
    
    Returns:
        tuple: (sunspot_database, sunspot_centers, suggested_threshold, cropped_solar_disk, center, radius)
    """
    print("=== SUNSPOT DETECTION PIPELINE ===")
    
    # Step 1: Load and preprocess image
    print("1. Loading and preprocessing image...")
    from DetectEllipse import detect_and_normalize_circle_ellipse
    
    cropped_solar_disk, (x_center, y_center), radius = detect_and_normalize_circle_ellipse(image_path)
    
    if cropped_solar_disk is None:
        print("ERROR: Solar disk detection failed!")
        return None, None, None, None, None, None
    
    print(f"   Solar disk detected: center=({x_center}, {y_center}), radius={radius}")
    
    # Step 2: Analyze image and suggest threshold
    print("2. Analyzing image and suggesting threshold...")
    suggested_threshold = suggest_optimal_threshold(cropped_solar_disk, (x_center, y_center), radius, show_diagnostics)
    
    # Step 3: Use manual threshold if provided, otherwise use suggested
    if manual_threshold is not None:
        final_threshold = manual_threshold
        print(f"3. Using manual threshold: {final_threshold}")
    else:
        final_threshold = suggested_threshold
        print(f"3. Using suggested threshold: {final_threshold}")
    
    # Step 4: Run detection with chosen threshold
    print("4. Running sunspot detection...")
    sunspot_database, sunspot_centers = run_detection_with_threshold(
        cropped_solar_disk, (x_center, y_center), radius, final_threshold, output_csv
    )
    
    # Step 5: Summary
    print("5. Detection complete!")
    if len(sunspot_database) > 0:
        spot_types = sunspot_database['spot_type'].value_counts()
        print(f"   Found {len(sunspot_centers)} sunspot regions: {dict(spot_types)}")
        print(f"   Database saved to: {output_csv}")
    else:
        print("   No sunspots detected with current threshold")
        print(f"   Try a higher threshold (lower intensity value) like {final_threshold + 20}")
    
    return sunspot_database, sunspot_centers, suggested_threshold, cropped_solar_disk, (x_center, y_center), radius


def suggest_optimal_threshold(solar_disk, center, radius, show_plot=False):
    """
    Analyze image and suggest an optimal threshold with minimal output.
    """
    height, width = solar_disk.shape
    x_center, y_center = center
    
    # Create inner mask
    y_coords, x_coords = np.ogrid[:height, :width]
    edge_buffer = int(radius * 0.05)
    inner_mask = (x_coords - x_center)**2 + (y_coords - y_center)**2 <= (radius - edge_buffer)**2
    disk_intensities = solar_disk[inner_mask]
    
    # Calculate candidate thresholds
    mean_intensity = disk_intensities.mean()
    std_intensity = disk_intensities.std()
    
    candidates = [
        int(np.percentile(disk_intensities, 1)),
        int(np.percentile(disk_intensities, 2)),
        int(mean_intensity - 3 * std_intensity),
        int(mean_intensity - 2.5 * std_intensity),
    ]
    
    # Filter valid candidates (should select 0.5-5% of pixels)
    valid_candidates = []
    for thresh in candidates:
        if thresh > 0:
            percentage = np.sum(disk_intensities < thresh) / len(disk_intensities) * 100
            if 0.5 <= percentage <= 5.0:
                valid_candidates.append((thresh, percentage))
    
    if valid_candidates:
        # Choose the most conservative valid threshold
        suggested = min(valid_candidates, key=lambda x: x[1])[0]
    else:
        # Fallback
        suggested = int(np.percentile(disk_intensities, 1.5))
    
    percentage = np.sum(disk_intensities < suggested) / len(disk_intensities) * 100
    print(f"   Image stats: mean={mean_intensity:.0f}, std={std_intensity:.0f}")
    print(f"   Suggested threshold: {suggested} (captures {percentage:.2f}% of disk)")
    
    if show_plot:
        show_threshold_analysis(solar_disk, inner_mask, suggested, disk_intensities)
    
    return suggested


def show_threshold_analysis(solar_disk, inner_mask, suggested_threshold, disk_intensities):
    """
    Show a compact threshold analysis plot.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Histogram
    axes[0].hist(disk_intensities, bins=50, alpha=0.7, edgecolor='black')
    axes[0].axvline(suggested_threshold, color='red', linestyle='--', 
                   label=f'Suggested: {suggested_threshold}')
    axes[0].set_xlabel('Intensity')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Intensity Distribution')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Original image
    axes[1].imshow(solar_disk, cmap='gray')
    axes[1].set_title('Original Solar Disk')
    axes[1].axis('off')
    
    # Threshold result
    binary = (solar_disk < suggested_threshold) & inner_mask
    result = solar_disk.copy()
    result[binary] = 0
    axes[2].imshow(result, cmap='gray')
    count = np.sum(binary)
    percentage = count / np.sum(inner_mask) * 100
    axes[2].set_title(f'Threshold {suggested_threshold}\n{count} pixels ({percentage:.2f}%)')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()


def run_detection_with_threshold(solar_disk, center, radius, threshold, output_csv):
    """
    Run the complete detection pipeline with a specific threshold.
    """
    height, width = solar_disk.shape
    x_center, y_center = center
    
    # Create masks
    y_coords, x_coords = np.ogrid[:height, :width]
    edge_buffer = int(radius * 0.05)
    inner_mask = (x_coords - x_center)**2 + (y_coords - y_center)**2 <= (radius - edge_buffer)**2
    
    # Find candidates
    candidates = (solar_disk < threshold) & inner_mask
    
    # Clean small regions
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(candidates.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    
    # Find connected regions
    labeled, num_regions = ndimage.label(cleaned, structure=ndimage.generate_binary_structure(2, 2))
    
    # Process regions
    valid_pixels = []
    background_estimate = estimate_solar_background_robust(solar_disk[inner_mask])
    
    for region_id in range(1, num_regions + 1):
        region_coords = np.where(labeled == region_id)
        region_pixels = list(zip(region_coords[0], region_coords[1]))
        region_size = len(region_pixels)
        
        # Basic size and position filtering
        if region_size >= 5:  # Minimum size
            # Get region center
            y_coords, x_coords = zip(*region_pixels)
            region_center_x = np.mean(x_coords)
            region_center_y = np.mean(y_coords)
            distance_from_center = np.sqrt((region_center_x - x_center)**2 + (region_center_y - y_center)**2)
            relative_distance = distance_from_center / radius
            
            # Position validation (exclude very edge regions)
            if relative_distance <= 0.9:
                # Intensity validation
                region_intensities = [solar_disk[y, x] for y, x in region_pixels]
                avg_intensity = np.mean(region_intensities)
                
                # Local contrast check
                local_contrast = compute_local_contrast_simple(solar_disk, region_pixels)
                
                # Classification and validation
                if local_contrast >= 8 and avg_intensity < threshold + 15:
                    # Classify by size
                    if region_size < 20:
                        spot_type = "small"
                    elif region_size < 100:
                        spot_type = "medium"
                    else:
                        spot_type = "large"
                    
                    # Add to valid pixels
                    for y, x in region_pixels:
                        valid_pixels.append((y, x, spot_type, region_id))
    
    # Create DataFrame
    if valid_pixels:
        pixel_data = []
        for y, x, spot_type, region_id in valid_pixels:
            distance_to_center = np.sqrt((x - x_center)**2 + (y - y_center)**2)
            intensity = solar_disk[y, x]
            pixel_data.append({
                "x": int(x), 
                "y": int(y), 
                "intensity": int(intensity),
                "distance_from_center": round(distance_to_center, 1),
                "relative_distance": round(distance_to_center / radius, 3),
                "spot_type": spot_type,
                "region_id": region_id
            })
        
        sunspot_database = pd.DataFrame(pixel_data)
        sunspot_database.to_csv(output_csv, index=False)
        
        # Create centers DataFrame
        sunspot_centers = group_pixels_into_centers(valid_pixels)
        
        # Show final result
        show_final_result(solar_disk, sunspot_centers, center, radius)
        
    else:
        sunspot_database = pd.DataFrame()
        sunspot_centers = pd.DataFrame()
    
    return sunspot_database, sunspot_centers


def compute_local_contrast_simple(image, region_pixels):
    """
    Simplified local contrast computation.
    """
    surrounding_pixels = []
    
    for y, x in region_pixels[:min(5, len(region_pixels))]:
        for dy in [-6, -3, 3, 6]:
            for dx in [-6, -3, 3, 6]:
                ny, nx = y + dy, x + dx
                if 0 <= ny < image.shape[0] and 0 <= nx < image.shape[1]:
                    surrounding_pixels.append(image[ny, nx])
    
    if len(surrounding_pixels) > 3:
        region_intensities = [image[y, x] for y, x in region_pixels]
        return np.mean(surrounding_pixels) - np.mean(region_intensities)
    else:
        return 0


def group_pixels_into_centers(valid_pixels):
    """
    Group pixels by region and calculate centers.
    """
    region_data = {}
    
    for y, x, spot_type, region_id in valid_pixels:
        if region_id not in region_data:
            region_data[region_id] = {
                'pixels': [],
                'spot_type': spot_type
            }
        region_data[region_id]['pixels'].append((x, y))
    
    centers = []
    for region_id, data in region_data.items():
        pixels = data['pixels']
        x_coords, y_coords = zip(*pixels)
        
        center_x = int(np.mean(x_coords))
        center_y = int(np.mean(y_coords))
        size = len(pixels)
        spot_type = data['spot_type']
        
        centers.append({
            'x_center': center_x,
            'y_center': center_y,
            'size': size,
            'spot_type': spot_type
        })
    
    return pd.DataFrame(centers)


def show_final_result(solar_disk, sunspot_centers, center, radius):
    """
    Show the final detection result with correct colors.
    """
    x_center, y_center = center
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Original
    axes[0].imshow(solar_disk, cmap='gray')
    axes[0].set_title('Original Solar Disk')
    circle = plt.Circle((x_center, y_center), radius, fill=False, color='green', linewidth=2)
    axes[0].add_patch(circle)
    axes[0].axis('off')
    
    # Final result
    final_image = cv2.cvtColor(solar_disk, cv2.COLOR_GRAY2BGR)
    
    if len(sunspot_centers) > 0:
        for _, row in sunspot_centers.iterrows():
            x, y = int(row['x_center']), int(row['y_center'])
            spot_type = row['spot_type']
            
            if spot_type == "small":
                color = (0, 255, 0)  # Green
                radius_dot = 3
            elif spot_type == "medium":
                color = (0, 255, 255)  # Yellow
                radius_dot = 4
            else:  # large
                color = (255, 0, 0)  # Blue in BGR
                radius_dot = 6
            
            cv2.circle(final_image, (x, y), radius_dot, color, -1)
    
    axes[1].imshow(final_image)
    axes[1].set_title('Detected Sunspots\n(Green=small, Yellow=medium, Blue=large)')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()


def estimate_solar_background_robust(intensities):
    """
    Robust estimation of solar background.
    """
    upper_quartile_mean = np.mean(intensities[intensities >= np.percentile(intensities, 75)])
    median_val = np.median(intensities)
    return np.mean([upper_quartile_mean, median_val * 1.1])


# Convenience functions for easy usage
def quick_detect(image_path, show_diagnostics=False):
    """
    Quick detection with automatic threshold.
    Returns: (sunspot_database, sunspot_centers, suggested_threshold, cropped_solar_disk, center, radius)
    """
    return detect_sunspots_complete_pipeline(image_path, show_diagnostics=show_diagnostics)


def detect_with_threshold(image_path, threshold, show_diagnostics=False):
    """
    Detection with manual threshold.
    Returns: (sunspot_database, sunspot_centers, suggested_threshold, cropped_solar_disk, center, radius)
    """
    return detect_sunspots_complete_pipeline(image_path, manual_threshold=threshold, 
                                           show_diagnostics=show_diagnostics)


def test_threshold(image_path, threshold):
    """
    Quick test of a specific threshold.
    """
    print(f"=== TESTING THRESHOLD {threshold} ===")
    
    from DetectEllipse import detect_and_normalize_circle_ellipse
    cropped_solar_disk, (x_center, y_center), radius = detect_and_normalize_circle_ellipse(image_path)
    
    if cropped_solar_disk is None:
        print("ERROR: Solar disk detection failed!")
        return
    
    # Quick test
    height, width = cropped_solar_disk.shape
    y_coords, x_coords = np.ogrid[:height, :width]
    edge_buffer = int(radius * 0.05)
    inner_mask = (x_coords - x_center)**2 + (y_coords - y_center)**2 <= (radius - edge_buffer)**2
    
    candidates = (cropped_solar_disk < threshold) & inner_mask
    num_candidates = np.sum(candidates)
    total_inner = np.sum(inner_mask)
    percentage = num_candidates / total_inner * 100
    
    print(f"Threshold {threshold} captures {num_candidates} pixels ({percentage:.2f}% of disk)")
    
    # Show result
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.imshow(cropped_solar_disk, cmap='gray')
    plt.title('Original')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    result = cropped_solar_disk.copy()
    result[candidates] = 0
    plt.imshow(result, cmap='gray')
    plt.title(f'Threshold {threshold}\n{num_candidates} pixels ({percentage:.2f}%)')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return percentage


# Example usage
if __name__ == "__main__":
    # Simple usage examples
    image_path = "path/to/your/image.jpg"
    
    # Method 1: Automatic threshold detection
    # db, centers, suggested = quick_detect(image_path, show_diagnostics=True)
    
    # Method 2: Test a specific threshold first
    # test_threshold(image_path, 100)
    
    # Method 3: Use manual threshold
    # db, centers, _ = detect_with_threshold(image_path, 85)
    
    print("Pipeline ready. Use one of the methods above with your image path.")