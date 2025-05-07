import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import display, HTML
from scipy.ndimage import binary_dilation
from skimage.measure import find_contours
from matplotlib.path import Path


def animate_storms(data, reflectivity_threshold=45, area_threshold=15, dilation_iterations=5, interval=100):
    """
    Animate storm detection over time from radar reflectivity data.

    Parameters:
    - data: np.ndarray of shape (T, H, W)
    - reflectivity_threshold: float, reflectivity threshold (dBZ) to identify storms
    - area_threshold: int, minimum number of pixels for a region to be considered a storm
    - dilation_iterations: int, number of binary dilation iterations to smooth/merge storms
    - interval: int, time between frames in ms

    Returns:
    - ani: matplotlib.animation.FuncAnimation object

    Example:
    >>> from storm_utils import animate_storms
    >>> from IPython.display import HTML, display
    >>> ani = animate_storms(data)
    >>> display(HTML(ani.to_jshtml()))

    """

    fig, ax = plt.subplots(figsize=(6, 7))
    cmap = plt.get_cmap("jet")

    img = ax.imshow(data[0], cmap=cmap, vmin=0, vmax=80)
    plt.colorbar(img, ax=ax, label="Reflectivity (dBZ)")
    title = ax.set_title("Radar Reflectivity - Time Step 0")

    storm_lines = []

    def update(frame):
        nonlocal storm_lines
        frame_data = data[frame]
        img.set_data(frame_data)
        title.set_text(f"Radar Reflectivity - Time Step {frame}")

        # Remove previous storm lines
        for line in storm_lines:
            line.remove()
        storm_lines = []

        # Apply threshold and dilation
        mask = frame_data > reflectivity_threshold
        dilated_mask = binary_dilation(mask, iterations=dilation_iterations)

        # Extract contours from binary mask
        contours = find_contours(dilated_mask.astype(float), 0.5)

        for contour in contours:
            path = Path(contour[:, ::-1])  # x, y
            xg, yg = np.meshgrid(np.arange(frame_data.shape[1]), np.arange(frame_data.shape[0]))
            coords = np.vstack((xg.ravel(), yg.ravel())).T
            inside = path.contains_points(coords).reshape(frame_data.shape)

            area = np.sum(mask & inside)

            if area >= area_threshold:
                line, = ax.plot(contour[:, 1], contour[:, 0], color='red', linewidth=2)
                storm_lines.append(line)

        return [img] + storm_lines

    plt.close(fig)  # Prevent duplicate static plot
    ani = animation.FuncAnimation(fig, update, frames=data.shape[0], interval=interval)
    return ani

def detect_storms(data, reflectivity_threshold=45, area_threshold=15, dilation_iterations=5):
    """
    Detects storms in radar data and returns their count and coordinates at each time step.

    Parameters:
    - data: ndarray of shape (T, H, W)
    - reflectivity_threshold: threshold for storm detection
    - area_threshold: minimum storm area
    - dilation_iterations: dilation iterations

    Returns:
    - List of dictionaries containing storm count and coordinates per frame.
    """
    results = []

    for t in range(data.shape[0]):
        frame_data = data[t]
        mask = frame_data > reflectivity_threshold
        dilated_mask = binary_dilation(mask, iterations=dilation_iterations)
        contours = find_contours(dilated_mask.astype(float), 0.5)

        storms_in_frame = []

        for contour in contours:
            path = Path(contour[:, ::-1])  # (x, y) ordering

            # Grid coordinates
            x_grid, y_grid = np.meshgrid(
                np.arange(frame_data.shape[1]), np.arange(frame_data.shape[0])
            )
            coords = np.vstack((x_grid.ravel(), y_grid.ravel())).T
            inside = path.contains_points(coords).reshape(frame_data.shape)

            area = np.sum(mask & inside)

            if area >= area_threshold:
                storm_coords = contour[:, [1, 0]].tolist()  # (x, y) points
                storms_in_frame.append({"mask": inside.astype(int), "contour": storm_coords})

        frame_result = {
            "time_step": t,
            "storm_count": len(storms_in_frame),
            "storm_coordinates": [s["contour"] for s in storms_in_frame],
            "storm_masks": [s["mask"] for s in storms_in_frame],
        }

        results.append(frame_result)

    return results

def detect_new_storm_formations(data, reflectivity_threshold=45, area_threshold=15, dilation_iterations=5, overlap_threshold=0.1):
    """
    Detects new storm formations: frames, count of newly formed storms, and their coordinates.

    Parameters:
    - data: np.ndarray of shape (T, H, W)
    - reflectivity_threshold: float, reflectivity threshold (dBZ) to identify storms
    - area_threshold: int, minimum number of pixels for a region to be considered a storm
    - dilation_iterations: int, number of binary dilation iterations to smooth/merge storms
    - overlap_threshold: float, maximum allowed overlap between a current storm and any previous storm for it to be considered "new".

    Returns:
    - new_storms_summary: list of dict
        Each dictionary contains:
            - 'time_step': int
            - 'new_storm_count': int
            - 'new_storm_coordinates': list of storm contour coordinates (list of (x, y))

    """
    storm_results = detect_storms(data, reflectivity_threshold, area_threshold, dilation_iterations)
    new_storms_summary = []

    previous_masks = []

    for frame_result in storm_results:
        new_count = 0
        new_coords = []
        current_masks = frame_result['storm_masks']
        current_coords = frame_result['storm_coordinates']

        for i, current in enumerate(current_masks):
            is_new = True
            for prev in previous_masks:
                overlap = np.sum(current & prev) / np.sum(current)
                if overlap > overlap_threshold:
                    is_new = False
                    break
            if is_new:
                new_count += 1
                new_coords.append(current_coords[i])

        new_storms_summary.append({
            "time_step": frame_result['time_step'],
            "new_storm_count": new_count,
            "new_storm_coordinates": new_coords
        })

        previous_masks = current_masks

    return new_storms_summary

def animate_new_storms(data, new_storms_result):
    """
    Animate the progression of newly formed storms over time using radar reflectivity data.

    Parameters:
    - data: np.ndarray of shape (T, H, W)
    - new_storms_result (list of dict): A list where each dict represents new storms at a specific time step.
        Each dict should have the keys:
        - "time_step" (int): Time step index.
        - "new_storm_count" (int): Number of new storms detected at that time step.
        - "new_storm_coordinates" (list of list of (x, y)): List of contours for each new storm.

    Returns:
    - ani: matplotlib.animation.FuncAnimation object

    Example:
    >>> from storm_utils import detect_new_storm_formations, animate_new_storms
    >>> from IPython.display import HTML, display            
    >>> new_storms = detect_new_storm_formations(test_data)
    >>> ani = animate_new_storms(test_data, new_storms)
    >>> display(HTML(ani.to_jshtml()))
    """

    fig, ax = plt.subplots(figsize=(6, 7))
    cmap = plt.get_cmap("jet")

    img = ax.imshow(data[0], cmap=cmap, vmin=0, vmax=80)
    title = ax.set_title("New Storms at Time Step 0")
    colorbar = plt.colorbar(img, ax=ax, label="Reflectivity (dBZ)")
    storm_lines = []

    def update(frame_id):
        nonlocal storm_lines
        frame = data[frame_id]
        img.set_data(frame)
        title.set_text(f"New Storms at Time Step {frame_id}")

        # Remove previous storm outlines
        for line in storm_lines:
            line.remove()
        storm_lines = []

        # Check if there are new storms at this frame
        frame_entry = next((f for f in new_storms_result if f["time_step"] == frame_id), None)
        if frame_entry and frame_entry["new_storm_count"] > 0:
            for contour in frame_entry["new_storm_coordinates"]:
                contour = np.array(contour)
                line, = ax.plot(contour[:, 0], contour[:, 1], color='lime', linewidth=2)
                storm_lines.append(line)

        return [img, title] + storm_lines
    plt.close(fig)
    ani = animation.FuncAnimation(fig, update, frames=data.shape[0], interval=200)
    return ani

def evaluate_new_storm_predictions(new_storms_pred, new_storms_true, overlap_threshold=0.2):
    """
    Compares predicted and true new storm initiations.

    Parameters:
    - new_storms_pred: list of dicts from detect_new_storm_formations(pred_max_cappi)
    - new_storms_true: list of dicts from detect_new_storm_formations(true_max_cappi)
    - overlap_threshold: float, minimum overlap ratio to count as a match

    Returns:
    - dict with counts:
        {
            "correct": int,  # matched at same time
            "early": int,    # matched one time step before
            "late": int,     # matched one time step after
            "false_positives": int,  # predicted storms with no match in ±1 time step
            "total_true": int,  # total true new storms
            "total_pred": int,  # total predicted new storms
            "correct_over_true": float,
            "correct_over_pred": float,
            "anytime_ratio": float,
            "false_positive_ratio": float,
        }
    """
    # Helper to get mask for a storm contour
    def contour_to_mask(contour, shape):
        from matplotlib.path import Path as MplPath
        # Ensure shape is a 2-tuple
        if len(shape) == 1:
            shape = (shape[0], 1)
        elif len(shape) == 0:
            shape = (1, 1)
        xg, yg = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        coords = np.vstack((xg.ravel(), yg.ravel())).T
        path = MplPath(np.array(contour))
        mask = path.contains_points(coords).reshape(shape)
        return mask

    # Build lookup: time_step -> list of (mask, used_flag)
    def build_storm_masks(new_storms, data_shape):
        lookup = {}
        for entry in new_storms:
            t = entry["time_step"]
            masks = []
            for contour in entry["new_storm_coordinates"]:
                mask = contour_to_mask(contour, data_shape)
                masks.append({"mask": mask, "used": False})
            lookup[t] = masks
        return lookup

    # Robustly infer data_shape from all contours in true and pred
    max_x, max_y = 0, 0
    found = False
    for storms in (new_storms_true, new_storms_pred):
        for entry in storms:
            for contour in entry["new_storm_coordinates"]:
                arr = np.array(contour)
                if arr.size == 0:
                    continue
                found = True
                if arr.ndim == 1 and arr.shape[0] == 2:
                    # Single point
                    x, y = arr[0], arr[1]
                    max_x = max(max_x, int(x))
                    max_y = max(max_y, int(y))
                elif arr.ndim == 2 and arr.shape[1] == 2:
                    # Contour
                    max_x = max(max_x, int(arr[:, 0].max()))
                    max_y = max(max_y, int(arr[:, 1].max()))
    if not found:
        return {"correct": 0, "early": 0, "late": 0, "false_positives": sum(e["new_storm_count"] for e in new_storms_pred), "total_true": 0, "total_pred": sum(e["new_storm_count"] for e in new_storms_pred), "correct_over_true": 0.0, "correct_over_pred": 0.0, "anytime_ratio": 0.0, "false_positive_ratio": 0.0}
    data_shape = (max_y + 1, max_x + 1)

    # Build mask lookups
    pred_lookup = build_storm_masks(new_storms_pred, data_shape)
    true_lookup = build_storm_masks(new_storms_true, data_shape)

    correct = 0
    early = 0
    late = 0
    matched_pred = set()  # (t, idx)
    matched_true = set()  # (t, idx)

    # For each true storm, look for matching pred storm in t-1, t, t+1
    for t, true_storms in true_lookup.items():
        for i, true_storm in enumerate(true_storms):
            found = False
            for dt, label in zip([0, -1, 1], ["correct", "early", "late"]):
                tt = t + dt
                if tt in pred_lookup:
                    for j, pred_storm in enumerate(pred_lookup[tt]):
                        if (tt, j) in matched_pred:
                            continue
                        # Compute overlap: intersection / area of true storm
                        intersection = np.sum(true_storm["mask"] & pred_storm["mask"])
                        area_true = np.sum(true_storm["mask"])
                        if area_true == 0:
                            continue
                        overlap = intersection / area_true
                        if overlap >= overlap_threshold:
                            if label == "correct":
                                correct += 1
                            elif label == "early":
                                early += 1
                            elif label == "late":
                                late += 1
                            matched_pred.add((tt, j))
                            matched_true.add((t, i))
                            found = True
                            break
                if found:
                    break

    # False positives: predicted storms not matched to any true storm in ±1 time step
    false_positives = 0
    for t, pred_storms in pred_lookup.items():
        for j, pred_storm in enumerate(pred_storms):
            if (t, j) in matched_pred:
                continue
            # Check if this pred storm matches any true storm in t-1, t, t+1
            matched = False
            for dt in [-1, 0, 1]:
                tt = t + dt
                if tt in true_lookup:
                    for i, true_storm in enumerate(true_lookup[tt]):
                        if (tt, i) in matched_true:
                            continue
                        intersection = np.sum(true_storm["mask"] & pred_storm["mask"])
                        area_pred = np.sum(pred_storm["mask"])
                        if area_pred == 0:
                            continue
                        overlap = intersection / area_pred
                        if overlap >= overlap_threshold:
                            matched = True
                            break
                if matched:
                    break
            if not matched:
                false_positives += 1

    total_true = sum(len(v) for v in true_lookup.values())
    total_pred = sum(len(v) for v in pred_lookup.values())

    # Ratios
    correct_over_true = correct / total_true if total_true > 0 else 0.0
    correct_over_pred = correct / total_pred if total_pred > 0 else 0.0
    anytime_ratio = (correct + early + late) / total_true if total_true > 0 else 0.0
    false_positive_ratio = false_positives / total_pred if total_pred > 0 else 0.0

    return {
        "correct": correct,
        "early": early,
        "late": late,
        "false_positives": false_positives,
        "total_true": total_true,
        "total_pred": total_pred,
        "correct_over_true": correct_over_true,
        "correct_over_pred": correct_over_pred,
        "anytime_ratio": anytime_ratio,
        "false_positive_ratio": false_positive_ratio,
    }

