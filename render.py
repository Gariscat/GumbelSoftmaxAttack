import matplotlib.pyplot as plt
import imageio
from tqdm import tqdm
import numpy as np

def events_to_combined_video(npzfile1, npzfile2, frame_size=(34, 34), time_window=1000, filename="combined_events.mp4", fps=120):
    """
    Convert two sequences of events from .npz files directly to a combined video file.
    
    Args:
        npzfile1 (str): Path to the first .npz file containing event data with 'x', 'y', 't', 'p' keys.
        npzfile2 (str): Path to the second .npz file containing event data with 'x', 'y', 't', 'p' keys.
        frame_size (tuple): Size of the frame (height, width), default is (34, 34) for NMNIST.
        time_window (int): Time interval for each frame in microseconds (us).
        filename (str): Output combined video filename.
        fps (int): Frames per second for the video.
    """
    
    # Extract events from each file
    x1, y1, t1, p1 = npzfile1['x'], npzfile1['y'], npzfile1['t'], npzfile1['p']
    x2, y2, t2, p2 = npzfile2['x'], npzfile2['y'], npzfile2['t'], npzfile2['p']
    
    # Combine into lists of tuples (x, y, t, p) for each sequence
    events1 = list(zip(x1, y1, t1, p1))
    events2 = list(zip(x2, y2, t2, p2))
    
    # Sort events by time
    events1 = sorted(events1, key=lambda e: e[2])  # Sort by t (time)
    events2 = sorted(events2, key=lambda e: e[2])  # Sort by t (time)
    
    # Get the time range for synchronization
    min_time = min(events1[0][2], events2[0][2])
    max_time = max(events1[-1][2], events2[-1][2])
    
    # Calculate the number of frames needed
    num_frames = (max_time - min_time) // time_window + 1
    
    # Create a video writer
    writer = imageio.get_writer(filename, fps=fps)
    
    # Initialize accumulated frames for both sequences
    accumulated_frame1 = np.zeros(frame_size, dtype=float)
    accumulated_frame2 = np.zeros(frame_size, dtype=float)
    
    # Generate frames and write to video
    for frame_idx in tqdm(range(num_frames), desc="Generating combined frames"):
        # Define the time window for the current frame
        start_time = min_time + frame_idx * time_window
        end_time = start_time + time_window
        
        # Process events for the current time window for both sequences
        for event in events1:
            x, y, t, p = event
            if start_time <= t < end_time:
                accumulated_frame1[y, x] += 1 if p > 0 else -1
        
        for event in events2:
            x, y, t, p = event
            if start_time <= t < end_time:
                accumulated_frame2[y, x] += 1 if p > 0 else -1
        
        # Normalize both frames for visualization
        normalized_frame1 = (accumulated_frame1 - accumulated_frame1.min()) / (accumulated_frame1.max() - accumulated_frame1.min() + 1e-6)
        normalized_frame2 = (accumulated_frame2 - accumulated_frame2.min()) / (accumulated_frame2.max() - accumulated_frame2.min() + 1e-6)
        
        # Apply a colormap for better contrast and informative visualization
        img1 = plt.cm.viridis(normalized_frame1)  # Apply plasma colormap
        img2 = plt.cm.viridis(normalized_frame2)  # Apply plasma colormap
        
        # Convert to uint8 and concatenate the two frames side-by-side
        img1 = (img1[:, :, :3] * 255).astype(np.uint8)  # Remove alpha channel and scale to 0-255
        img2 = (img2[:, :, :3] * 255).astype(np.uint8)  # Remove alpha channel and scale to 0-255
        combined_frame = np.hstack((img1, img2))
        
        # Concatenate the two frames side-by-side
        combined_frame = np.hstack((img1, img2))
        
        # Append combined frame to video
        writer.append_data(combined_frame)
    
    writer.close()
    print(f"Combined video saved as {filename}")

# Example usage
# events_to_combined_video("path_to_file1.npz", "path_to_file2.npz", frame_size=(34, 34), time_window=1000, filename="combined_events.mp4", fps=10)

# import numpy as np
# import imageio
# import matplotlib.pyplot as plt
# from tqdm import tqdm

def events_to_combined_video_unsynced(npzfile1, npzfile2, frame_size=(34, 34), time_window=1000, filename="unsynced_combined_events.mp4", fps=120):
    """
    Create a combined video from two unsynchronized event sequences by aligning events based on time windows.
    
    Args:
        npzfile1 (str): Path to the first .npz file containing event data with 'x', 'y', 't', 'p' keys.
        npzfile2 (str): Path to the second .npz file containing event data with 'x', 'y', 't', 'p' keys.
        frame_size (tuple): Size of the frame (height, width), default is (34, 34) for NMNIST.
        time_window (int): Time interval for each frame in microseconds (us).
        filename (str): Output combined video filename.
        fps (int): Frames per second for the video.
    """
    # Load the .npz files
    # data1 = np.load(npzfile1)
    # data2 = np.load(npzfile2)
    data1 = npzfile1
    data2 = npzfile2
    
    # Extract events from each file
    x1, y1, t1, p1 = data1['x'], data1['y'], data1['t'], data1['p']
    x2, y2, t2, p2 = data2['x'], data2['y'], data2['t'], data2['p']
    
    # Combine into lists of tuples (x, y, t, p) for each sequence
    events1 = list(zip(x1, y1, t1, p1))
    events2 = list(zip(x2, y2, t2, p2))
    
    # Sort events by time
    events1 = sorted(events1, key=lambda e: e[2])  # Sort by t (time)
    events2 = sorted(events2, key=lambda e: e[2])  # Sort by t (time)
    
    # Get the time range for synchronization
    min_time = min(events1[0][2], events2[0][2])
    max_time = max(events1[-1][2], events2[-1][2])
    
    # Calculate the number of frames needed
    num_frames = (max_time - min_time) // time_window + 1
    
    # Create a video writer
    writer = imageio.get_writer(filename, fps=fps)
    
    # Initialize accumulated frames for both sequences
    accumulated_frame1 = np.zeros(frame_size, dtype=float)
    accumulated_frame2 = np.zeros(frame_size, dtype=float)
    
    # Generate frames and write to video
    for frame_idx in tqdm(range(num_frames), desc="Generating combined frames"):
        # Define the time window for the current frame
        start_time = min_time + frame_idx * time_window
        end_time = start_time + time_window
        
        # Clear frames for current time window
        frame1 = np.zeros(frame_size, dtype=float)
        frame2 = np.zeros(frame_size, dtype=float)
        
        # Process events for the current time window for the first sequence
        for event in events1:
            x, y, t, p = event
            if start_time <= t < end_time:
                frame1[y, x] += 1 if p > 0 else -1
        
        # Process events for the current time window for the second sequence
        for event in events2:
            x, y, t, p = event
            if start_time <= t < end_time:
                frame2[y, x] += 1 if p > 0 else -1
        
        # Normalize both frames for visualization
        normalized_frame1 = (frame1 - frame1.min()) / (frame1.max() - frame1.min() + 1e-6)
        normalized_frame2 = (frame2 - frame2.min()) / (frame2.max() - frame2.min() + 1e-6)
        
        # Apply a colormap for better contrast and informative visualization
        img1 = plt.cm.viridis(normalized_frame1) * 255  # Apply plasma colormap and scale to 0-255
        img2 = plt.cm.viridis(normalized_frame2) * 255  # Apply plasma colormap and scale to 0-255
        
        # Remove alpha channel by taking only RGB values (0:3)
        img1 = img1[:, :, :3].astype(np.uint8)
        img2 = img2[:, :, :3].astype(np.uint8)
        
        # Concatenate the two frames side-by-side
        combined_frame = np.hstack((img1, img2))
        
        # Append combined frame to video
        writer.append_data(combined_frame)
    
    writer.close()
    print(f"Combined video saved as {filename}")

# Example usage
# events_to_combined_video_unsynced("path_to_file1.npz", "path_to_file2.npz", frame_size=(34, 34), time_window=1000, filename="unsynced_combined_events.mp4", fps=10)

def events_to_separate_videos(npzfile1, npzfile2, frame_size=(34, 34), time_window=1000, filenames=("events1.mp4", "events2.mp4"), fps=120, use_gif=False):
    """
    Convert two sequences of events from .npz files to separate video files.
    
    Args:
        npzfile1 (str): Path to the first .npz file containing event data with 'x', 'y', 't', 'p' keys.
        npzfile2 (str): Path to the second .npz file containing event data with 'x', 'y', 't', 'p' keys.
        frame_size (tuple): Size of the frame (height, width), default is (34, 34) for NMNIST.
        time_window (int): Time interval for each frame in microseconds (us).
        filenames (tuple): Output filenames for the two separate videos.
        fps (int): Frames per second for the video.
    """

    def process_events(npzfile, frame_size, time_window, filename, fps, use_gif):
        """Helper function to process events from one file and save as video."""
        if use_gif:
            filename = filename.replace(".mp4", ".gif")
            writer = imageio.get_writer(filename, fps=fps, mode='I')
        else:
            writer = imageio.get_writer(filename, fps=fps)
        
        x, y, t, p = npzfile['x'], npzfile['y'], npzfile['t'], npzfile['p']
        events = list(zip(x, y, t, p))
        events = sorted(events, key=lambda e: e[2])  # Sort by time
        
        min_time = events[0][2]
        max_time = events[-1][2]
        num_frames = (max_time - min_time) // time_window + 1

        
        accumulated_frame = np.zeros(frame_size, dtype=float)

        for frame_idx in tqdm(range(num_frames), desc=f"Generating frames for {filename}"):
            start_time = min_time + frame_idx * time_window
            end_time = start_time + time_window

            for event in events:
                x, y, t, p = event
                if start_time <= t < end_time:
                    accumulated_frame[y, x] += 1 if p > 0 else -1

            normalized_frame = (accumulated_frame - accumulated_frame.min()) / (accumulated_frame.max() - accumulated_frame.min() + 1e-6)
            img = plt.cm.viridis(normalized_frame)
            img = (img[:, :, :3] * 255).astype(np.uint8)  # Remove alpha channel and scale to 0-255
            writer.append_data(img)

        writer.close()
        print(f"Video saved as {filename}")

    # Process each event sequence and save as separate videos
    process_events(npzfile1, frame_size, time_window, filenames[0], fps, use_gif)
    process_events(npzfile2, frame_size, time_window, filenames[1], fps, use_gif)


def events_to_combined_video_normalized(npzfile1, npzfile2, frame_size=(34, 34), num_frames=100, filename="normalized_combined_events.mp4", fps=10):
    """
    Create a combined video from two unsynchronized event sequences by normalizing timestamps.
    
    Args:
        npzfile1 (str): Path to the first .npz file containing event data with 'x', 'y', 't', 'p' keys.
        npzfile2 (str): Path to the second .npz file containing event data with 'x', 'y', 't', 'p' keys.
        frame_size (tuple): Size of the frame (height, width), default is (34, 34) for NMNIST.
        num_frames (int): Total number of frames to render.
        filename (str): Output combined video filename.
        fps (int): Frames per second for the video.
    """
    # Load the .npz files
    # data1 = np.load(npzfile1)
    # data2 = np.load(npzfile2)
    data1 = npzfile1
    data2 = npzfile2
    
    # Extract events from each file
    x1, y1, t1, p1 = data1['x'], data1['y'], data1['t'], data1['p']
    x2, y2, t2, p2 = data2['x'], data2['y'], data2['t'], data2['p']
    
    # Combine into lists of tuples (x, y, t, p) for each sequence
    events1 = list(zip(x1, y1, t1, p1))
    events2 = list(zip(x2, y2, t2, p2))
    
    # Find the min and max times for normalization
    min_time1, max_time1 = min(t1), max(t1)
    min_time2, max_time2 = min(t2), max(t2)
    
    # Overall min and max times across both sequences
    min_time = min(min_time1, min_time2)
    max_time = max(max_time1, max_time2)
    
    # Normalize timestamps to a range of 0 to num_frames-1
    def normalize_time(events, min_time, max_time, num_frames):
        return [(x, y, int((t - min_time) / (max_time - min_time) * (num_frames - 1)), p) for x, y, t, p in events]

    events1 = normalize_time(events1, min_time, max_time, num_frames)
    events2 = normalize_time(events2, min_time, max_time, num_frames)
    
    # Create a video writer
    writer = imageio.get_writer(filename, fps=fps)
    
    # Initialize accumulated frames for both sequences
    accumulated_frame1 = np.zeros(frame_size, dtype=float)
    accumulated_frame2 = np.zeros(frame_size, dtype=float)
    
    # Generate frames and write to video
    for frame_idx in tqdm(range(num_frames), desc="Generating normalized frames"):
        # Clear frames for current frame
        frame1 = np.zeros(frame_size, dtype=float)
        frame2 = np.zeros(frame_size, dtype=float)
        
        # Process events for the current frame index for the first sequence
        for event in events1:
            x, y, t, p = event
            if t == frame_idx:  # Match the current normalized time frame
                frame1[y, x] += 1 if p > 0 else -1
        
        # Process events for the current frame index for the second sequence
        for event in events2:
            x, y, t, p = event
            if t == frame_idx:  # Match the current normalized time frame
                frame2[y, x] += 1 if p > 0 else -1
        
        # Normalize both frames for visualization
        normalized_frame1 = (frame1 - frame1.min()) / (frame1.max() - frame1.min() + 1e-6)
        normalized_frame2 = (frame2 - frame2.min()) / (frame2.max() - frame2.min() + 1e-6)
        
        # Apply a colormap for better contrast and informative visualization
        img1 = plt.cm.viridis(normalized_frame1) * 255  # Apply plasma colormap and scale to 0-255
        img2 = plt.cm.viridis(normalized_frame2) * 255  # Apply plasma colormap and scale to 0-255
        
        # Remove alpha channel by taking only RGB values (0:3)
        img1 = img1[:, :, :3].astype(np.uint8)
        img2 = img2[:, :, :3].astype(np.uint8)
        
        # Concatenate the two frames side-by-side
        combined_frame = np.hstack((img1, img2))
        
        # Append combined frame to video
        writer.append_data(combined_frame)
    
    writer.close()
    print(f"Combined video saved as {filename}")

# Example usage
# events_to_combined_video_normalized("path_to_file1.npz", "path_to_file2.npz", frame_size=(34, 34), num_frames=100, filename="normalized_combined_events.mp4", fps=10)
