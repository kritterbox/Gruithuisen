"""
Interactive Streamlit Application for Sunspot Detection
Run with: streamlit run sunspot_app.py
"""

import streamlit as st
import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import tempfile
import os
from PIL import Image

# Import your modules
from DetectEllipse import detect_and_normalize_circle_ellipse
from consolidated_sunspot_pipeline import (
    quick_detect, 
    detect_with_threshold, 
    test_threshold,
    suggest_optimal_threshold,
    run_detection_with_threshold
)

def main():
    st.set_page_config(
        page_title="Sunspot Detection Tool",
        page_icon="☀️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("☀️ Interactive Sunspot Detection Tool")
    st.markdown("Upload a solar disk image and interactively adjust the detection threshold")
    
    # Sidebar for controls
    st.sidebar.header("🔧 Controls")
    
    # File upload
    uploaded_file = st.sidebar.file_uploader(
        "Choose a solar disk image",
        type=['png', 'jpg', 'jpeg', 'tiff', 'bmp'],
        help="Upload an image of a solar disk"
    )
    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            temp_path = tmp_file.name
        
        try:
            # Process the image
            process_image(temp_path, uploaded_file.name)
        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    else:
        st.info("👆 Please upload a solar disk image to begin detection")
        
        # Show example workflow
        st.markdown("### 📋 How to use this tool:")
        st.markdown("""
        1. **Upload** a solar disk image using the file uploader in the sidebar
        2. **Review** the automatic preprocessing and suggested threshold  
        3. **Adjust** the detection threshold using the interactive slider
        4. **Download** the results as CSV files
        5. **Analyze** the detected sunspots with the summary statistics
        """)


def process_image(image_path, filename):
    """Process the uploaded image and show interactive detection."""
    
    st.header(f"🖼️ Processing: {filename}")
    
    # Step 1: Initial processing with progress bar
    with st.spinner("Processing solar disk..."):
        try:
            # Initial detection to get suggested threshold
            result = quick_detect(image_path, show_diagnostics=False)
            if result[0] is None:  # Check if detection failed
                st.error("❌ Solar disk detection failed. Please check your image.")
                return
                
            initial_db, initial_centers, suggested_threshold, cropped_solar_disk, center, radius = result
            
        except Exception as e:
            st.error(f"❌ Error processing image: {str(e)}")
            return
    
    # Store data in session state for persistence
    st.session_state.cropped_solar_disk = cropped_solar_disk
    st.session_state.center = center
    st.session_state.radius = radius
    st.session_state.suggested_threshold = suggested_threshold
    st.session_state.image_path = image_path
    
    # Show basic image info in a compact way
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Size", f"{cropped_solar_disk.shape[1]}×{cropped_solar_disk.shape[0]}")
    with col2:
        st.metric("Radius", f"{radius}px")
    with col3:
        st.metric("Suggested", suggested_threshold)
    
    # Show original image compactly
    st.subheader("📸 Preprocessed Solar Disk")
    
    # Interactive detection section - combine everything
    st.header("🔍 Interactive Sunspot Detection")
    
    # Move threshold slider to sidebar for better visibility
    st.sidebar.header("🎛️ Detection Parameters")
    
    min_threshold = max(10, suggested_threshold - 50)
    max_threshold = min(255, suggested_threshold + 100)
    
    threshold = st.sidebar.slider(
        "Detection Threshold",
        min_value=min_threshold,
        max_value=max_threshold,
        value=suggested_threshold,
        step=1,
        help=f"Lower = more spots, Higher = fewer spots. Suggested: {suggested_threshold}"
    )
    
    # Calculate threshold impact for sidebar feedback
    x_center, y_center = center  # Extract coordinates from center tuple
    height, width = cropped_solar_disk.shape
    y_coords, x_coords = np.ogrid[:height, :width]
    edge_buffer = int(radius * 0.05)
    inner_mask = (x_coords - x_center)**2 + (y_coords - y_center)**2 <= (radius - edge_buffer)**2
    
    candidates = (cropped_solar_disk < threshold) & inner_mask
    num_candidates = np.sum(candidates)
    total_pixels = np.sum(inner_mask)
    percentage = num_candidates / total_pixels * 100
    
    st.sidebar.metric("Candidate Pixels", f"{num_candidates} ({percentage:.2f}%)")
    
    # Threshold feedback in sidebar
    if percentage < 0.5:
        st.sidebar.info("🔵 Very conservative")
    elif percentage < 2:
        st.sidebar.success("✅ Good balance")
    elif percentage < 5:
        st.sidebar.warning("🟡 Moderate")
    else:
        st.sidebar.error("🔴 Very permissive")
    
    # Run detection with current threshold
    with st.spinner("Detecting sunspots..."):
        try:
            db, centers, _, _, _, _ = detect_with_threshold(image_path, threshold, show_diagnostics=False)
        except Exception as e:
            st.error(f"Detection failed: {str(e)}")
            db, centers = pd.DataFrame(), pd.DataFrame()
    
    # Show all three images in a row
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("📸 Original")
        fig, ax = plt.subplots(1, 1, figsize=(3.5, 3.5))
        ax.imshow(cropped_solar_disk, cmap='gray')
        ax.set_title('Solar Disk')
        circle = plt.Circle((x_center, y_center), radius, fill=False, color='green', linewidth=1)
        ax.add_patch(circle)
        ax.axis('off')
        st.pyplot(fig)
        plt.close()
    
    with col2:
        st.subheader("🎯 Threshold Preview")
        fig, ax = plt.subplots(1, 1, figsize=(3.5, 3.5))
        result = cropped_solar_disk.copy()
        result[candidates] = 0
        ax.imshow(result, cmap='gray')
        ax.set_title(f'Threshold {threshold}')
        ax.axis('off')
        st.pyplot(fig)
        plt.close()
    
    with col3:
        st.subheader("🔍 Final Detection")
        if len(db) > 0:
            fig = create_detection_plot_compact(cropped_solar_disk, centers, center, radius)
            st.pyplot(fig)
            plt.close()
            
            # Quick stats below the image
            spot_types = db['spot_type'].value_counts()
            st.write(f"**{len(centers)} regions**")
            st.write(f"S:{spot_types.get('small', 0)} M:{spot_types.get('medium', 0)} L:{spot_types.get('large', 0)}")
        else:
            fig, ax = plt.subplots(1, 1, figsize=(3.5, 3.5))
            ax.imshow(cropped_solar_disk, cmap='gray')
            ax.set_title('No spots detected')
            ax.axis('off')
            st.pyplot(fig)
            plt.close()
            st.warning("No spots - try lower threshold")
    
    # Detailed results in expandable sections (only if spots detected)
    if len(db) > 0:
        show_detailed_results(db, centers, threshold)


def create_detection_plot_compact(solar_disk, sunspot_centers, center, radius):
    """Create a compact detection visualization plot."""
    x_center, y_center = center
    
    fig, ax = plt.subplots(1, 1, figsize=(3.5, 3.5))
    
    # Convert to color for annotation
    annotated = cv2.cvtColor(solar_disk, cv2.COLOR_GRAY2BGR)
    
    # Draw sunspot centers with correct colors
    for _, row in sunspot_centers.iterrows():
        x, y = int(row['x_center']), int(row['y_center'])
        spot_type = row['spot_type']
        
        if spot_type == "small":
            color = (0, 255, 0)  # Green in BGR
            radius_dot = 2
        elif spot_type == "medium":
            color = (0, 255, 255)  # Yellow in BGR
            radius_dot = 3
        else:  # large
            color = (255, 0, 0)  # Blue in BGR
            radius_dot = 4
        
        cv2.circle(annotated, (x, y), radius_dot, color, -1)
    
    # Convert BGR back to RGB for matplotlib
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    
    ax.imshow(annotated_rgb)
    ax.set_title('Detected Sunspots')
    ax.axis('off')
    
    return fig


def show_detailed_results(sunspot_db, sunspot_centers, threshold):
    """Show detailed results in expandable sections."""
    
    st.header("📊 Detailed Results")
    
    # Summary statistics in a compact format
    with st.expander("📈 Summary Statistics", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Regions", len(sunspot_centers))
            if len(sunspot_centers) > 0:
                sizes = sunspot_centers['size']
                st.metric("Avg Size", f"{sizes.mean():.0f} px")
        
        with col2:
            spot_types = sunspot_db['spot_type'].value_counts()
            st.metric("Small Spots", spot_types.get('small', 0))
            st.metric("Medium Spots", spot_types.get('medium', 0))
        
        with col3:
            st.metric("Large Spots", spot_types.get('large', 0))
            st.metric("Darkest Pixel", sunspot_db['intensity'].min())
    
    # Size distribution chart
    with st.expander("📊 Size Distribution", expanded=False):
        if len(sunspot_centers) > 0:
            fig, ax = plt.subplots(1, 1, figsize=(8, 3))
            sizes = sunspot_centers['size']
            ax.hist(sizes, bins=min(15, len(sizes)), alpha=0.7, edgecolor='black')
            ax.set_xlabel('Spot Size (pixels)')
            ax.set_ylabel('Count')
            ax.set_title('Distribution of Sunspot Sizes')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close()
    
    # Data tables
    with st.expander("📋 Sunspot Centers Data", expanded=False):
        st.dataframe(
            sunspot_centers,
            use_container_width=True,
            column_config={
                "x_center": st.column_config.NumberColumn("X", format="%d"),
                "y_center": st.column_config.NumberColumn("Y", format="%d"), 
                "size": st.column_config.NumberColumn("Size", format="%d"),
                "spot_type": st.column_config.TextColumn("Type")
            }
        )
    
    with st.expander("🗃️ Full Pixel Database (first 100 rows)", expanded=False):
        display_db = sunspot_db.head(100)
        st.dataframe(display_db, use_container_width=True)
        if len(sunspot_db) > 100:
            st.info(f"Showing 100 of {len(sunspot_db)} total pixels")
    
    # Download section - always visible
    st.subheader("💾 Download Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        csv_centers = sunspot_centers.to_csv(index=False)
        st.download_button(
            label="📄 Download Centers CSV",
            data=csv_centers,
            file_name=f"sunspot_centers_t{threshold}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col2:
        csv_db = sunspot_db.to_csv(index=False)
        st.download_button(
            label="📊 Download Full Database CSV",
            data=csv_db,
            file_name=f"sunspot_database_t{threshold}.csv",
            mime="text/csv",
            use_container_width=True
        )


def show_threshold_impact(threshold):
    """Show the impact of the current threshold setting - REMOVED (now integrated in main layout)."""
    pass


def show_detection_results(sunspot_db, sunspot_centers, threshold):
    """Display the detection results - REMOVED (now integrated in main layout).""" 
    pass


def show_summary_statistics(sunspot_db, sunspot_centers):
    """Show detailed summary statistics - REMOVED (now in expandable sections)."""
    pass


if __name__ == "__main__":
    main()