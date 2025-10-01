# ==============================================================================
# Perception Assignment: Cuboid Rotation Analysis
#
# This script processes depth data from a ROS bag to estimate the rotation
# of a cuboid. It includes two methods for calculating the rotation axis
# for comparison and generates all required outputs and visualizations.
# ==============================================================================

# --- Step 1: Imports ---
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

# Analysis Libraries
from rosbags.highlevel import AnyReader
from sklearn.linear_model import RANSACRegressor
from sklearn.decomposition import PCA
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

# --- Step 2: Helper Functions ---

def ros_image_to_numpy(msg):
    """Converts a ROS sensor_msgs/Image to a NumPy array, supporting 16UC1."""
    if msg.encoding == '16UC1':
        return np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width)
    raise NotImplementedError(f"Encoding '{msg.encoding}' is not supported.")

def depth_to_point_cloud(depth_image, intrinsics):
    """Converts a depth image to a 3D point cloud, converting mm to meters."""
    fx, fy, cx, cy = intrinsics
    points = []
    height, width = depth_image.shape
    for v in range(height):
        for u in range(width):
            d = depth_image[v, u]
            if d > 0:
                z = d / 1000.0  # Convert mm to meters
                x = (u - cx) * z / fx
                y = (v - cy) * z / fy
                points.append([x, y, z])
    return np.array(points)

# --- Step 3: Core Processing Function ---

def process_ros_bag(bag_path):
    """Loads and processes all frames from the ROS bag."""
    # Load data from bag
    depth_images, timestamps = [], []
    try:
        with AnyReader([bag_path]) as reader:
            connections = [x for x in reader.connections if x.topic == '/depth']
            for conn, ts, raw in reader.messages(connections=connections):
                msg = reader.deserialize(raw, conn.msgtype)
                depth_images.append(ros_image_to_numpy(msg))
                timestamps.append(ts)
        print(f"✅ Successfully loaded {len(depth_images)} images.")
    except Exception as e:
        print(f"❌ Error reading ROS bag: {e}")
        return None, None, None

    # Process each frame
    all_face_data, all_face_points = [], []
    height, width = depth_images[0].shape
    intrinsics = (width / 2.0, width / 2.0, width / 2.0, height / 2.0)

    for i, depth_image in enumerate(depth_images):
        point_cloud = depth_to_point_cloud(depth_image, intrinsics)
        
        if point_cloud.shape[0] > 0: # Filter background
            median_z = np.median(point_cloud[:, 2])
            point_cloud = point_cloud[np.abs(point_cloud[:, 2] - median_z) < 0.25]

        if point_cloud.shape[0] < 20: # Check for enough points
            all_face_data.append(None); all_face_points.append(None); continue

        # Find face with RANSAC
        ransac = RANSACRegressor(residual_threshold=0.01).fit(point_cloud[:, 0:2], point_cloud[:, 2])
        face_points = point_cloud[ransac.inlier_mask_]

        if len(face_points) < 20: # Check if RANSAC found a face
            all_face_data.append(None); all_face_points.append(None); continue
            
        # Calculate properties
        a, b = ransac.estimator_.coef_
        normal = np.array([a, b, -1.0]); normal /= np.linalg.norm(normal)
        if normal[2] > 0: normal = -normal
        dot_prod = np.clip(np.dot(normal, np.array([0, 0, -1.0])), -1.0, 1.0)
        angle = np.rad2deg(np.arccos(dot_prod))
        area = ConvexHull(face_points[:, 0:2]).volume

        all_face_points.append(face_points)
        all_face_data.append({"normal": normal, "angle_deg": angle, "area_m2": area})
        
    print("✅ Successfully processed all frames.")
    return all_face_data, all_face_points, timestamps

# --- Step 4: Analysis Functions (Both Approaches) ---

def analyze_rotation_axis_simple(valid_face_data):
    """Approach 1: Averages the cross-product of consecutive normals."""
    axes = []
    for i in range(len(valid_face_data) - 1):
        n1, n2 = valid_face_data[i]["normal"], valid_face_data[i+1]["normal"]
        axis = np.cross(n1, n2)
        if np.linalg.norm(axis) > 1e-6:
            axes.append(axis / np.linalg.norm(axis))
    return np.mean(axes, axis=0) if axes else np.array([0,0,0])

def analyze_rotation_axis_pca(valid_face_data):
    """Approach 2: Uses PCA on all normals for a more robust estimate."""
    if len(valid_face_data) < 3: return np.array([0,0,0])
    normals = [d['normal'] for d in valid_face_data]
    pca = PCA(n_components=3).fit(normals)
    # The axis is the component with the smallest variance
    return pca.components_[2] / np.linalg.norm(pca.components_[2])

# --- Step 5: Visualization Functions ---

def create_box_animation(all_face_data, all_face_points, axis, filename="box_rotation.gif"):
    """Creates and saves a 3D animation of the rotating box."""
    valid_indices = [i for i, d in enumerate(all_face_data) if d]
    valid_data = [d for d in all_face_data if d]
    
    # Estimate box model
    best_idx = valid_indices[np.argmin([d['angle_deg'] for d in valid_data])]
    best_pts = all_face_points[best_idx]
    w, h = np.ptp(best_pts[:,0]), np.ptp(best_pts[:,1])
    d = (w+h)/2.0
    stable_center = np.mean([np.mean(all_face_points[i], 0) for i in valid_indices], 0)
    
    verts = np.array([[-.5,-.5,-.5],[.5,-.5,-.5],[.5,.5,-.5],[-.5,.5,-.5],[-.5,-.5,.5],[.5,-.5,.5],[.5,.5,.5],[-.5,.5,.5]]) * [w,h,d]
    edges = [[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]]
    
    # Setup plot
    fig = plt.figure(figsize=(8, 6)); ax = fig.add_subplot(111, projection='3d')
    lines = [ax.plot([],[],[], 'b-')[0] for _ in edges]
    
    def update(frame):
        if all_face_data[frame] is None:
            for line in lines: line.set_data_3d([],[],[])
            ax.set_title(f'Frame {frame+1} (No Face Detected)')
            return lines
        
        rot, _ = Rotation.align_vectors([all_face_data[frame]['normal']], [np.array([0,0,-1.])])
        pos_verts = rot.apply(verts) + stable_center
        for i, edge in enumerate(edges):
            lines[i].set_data_3d(pos_verts[edge,0], pos_verts[edge,1], pos_verts[edge,2])
        ax.set_title(f'Frame {frame+1}')
        return lines

    # Set consistent plot limits
    all_pts = np.concatenate([p for p in all_face_points if p is not None])
    c, r = (np.min(all_pts,0)+np.max(all_pts,0))/2, np.max(np.ptp(all_pts,0))*1.0
    ax.set_xlim(c[0]-r,c[0]+r); ax.set_ylim(c[1]-r,c[1]+r); ax.set_zlim(c[2]-r,c[2]+r)
    ax.set_xlabel('X(m)'), ax.set_ylabel('Y(m)'), ax.set_zlabel('Z(m)')
    
    # Create and save animation
    ani = FuncAnimation(fig, update, frames=len(all_face_data), interval=500, blit=False)
    ani.save(filename, writer='imageio', fps=2)
    plt.close()
    print(f"✅ Animation saved to '{filename}'")

# --- Step 6: Main Execution ---

def main():
    """Main function to run the entire analysis pipeline."""
    bag_path = Path('.') # Assumes bag files are in the current directory
    
    all_face_data, all_face_points, timestamps = process_ros_bag(bag_path)
    
    if all_face_data is None:
        print("❌ Processing failed. Exiting.")
        return

    valid_face_data = [d for d in all_face_data if d]

    # --- Compare both approaches for rotation axis ---
    axis_simple = analyze_rotation_axis_simple(valid_face_data)
    axis_pca = analyze_rotation_axis_pca(valid_face_data)
    
    print("\n--- Rotation Axis Analysis ---")
    print(f"Approach 1 (Simple Avg): {np.array2string(axis_simple, precision=4)}")
    print(f"Approach 2 (Robust PCA): {np.array2string(axis_pca, precision=4)}")

    # --- Print final results table ---
    print("\n--- Final Results Table ---")
    print("Image | Normal Angle (deg) | Visible Area (m^2)")
    print("--------------------------------------------------")
    for i, data in enumerate(all_face_data):
        if data:
            print(f"{i+1:5d} | {data['angle_deg']:19.2f} | {data['area_m2']:18.4f}")
        else:
            print(f"{i+1:5d} | {'--':>19} | {'--':>18}")
            
    # --- Save final deliverables ---
    final_axis = axis_pca # Choose the best algorithm for the final output
    np.savetxt('rotation_axis.txt', final_axis, header='[X, Y, Z]')
    print("\n✅ Final axis (from PCA) saved to 'rotation_axis.txt'")
    
    # --- Generate visualizations ---
    create_box_animation(all_face_data, all_face_points, final_axis)

if __name__ == "__main__":
    main()