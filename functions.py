import numpy as np
import cv2
import copy
from scipy.spatial import ConvexHull

def resize_image_and_label(image, label, length = 64):
    # shift label
    if image.shape[2] == 4:  # Image has 4 channels (RGBA)
        # Convert to RGB by ignoring the alpha channel
        image = image[:, :, :3]
    # make the image a square
    Height = np.shape(image)[0]
    Width = np.shape(image)[1]
    Area = Height*Width
    diff = abs(Width - Height)
    # Determine padding for width and height
    if Width > Height:
        # Width is greater than height, pad top and bottom
        top_pad = diff// 2
        bottom_pad = diff - top_pad
        left_pad = 0
        right_pad = 0
    else:
        # Height is greater than width, pad left and right
        left_pad = diff // 2
        top_pad = 0
        bottom_pad = 0
        right_pad = diff - left_pad
    # Apply padding
    image = cv2.copyMakeBorder(image, top_pad, bottom_pad, left_pad, right_pad, cv2.BORDER_CONSTANT, value=[255, 255, 255])
    
    if diff > 0:
        label = np.copy(label)
        for i in range(len(label)):
            # Adjusting x coordinates for left padding
            label[i][0] += left_pad
            # Adjusting y coordinates for top padding
            label[i][1] += top_pad
    # rescaled dimensions
    Height = np.shape(image)[0]
    Width = np.shape(image)[1]
    Area = Height*Width
    area = length*length
    width  = int(np.sqrt(area/Area) * Width)
    height = int(np.sqrt(area/Area) * Height)
    image = cv2.resize(image, (length,length))
    label = label * np.sqrt(area/Area)
    for l in label:
        for i in l:
            i = int(i)
    
    return image, label

def adjust_tracked_label(label,i):
    #1080x1920
    tracked_index = [i for i in range(489,643)]
    for j in range(757,936):
        tracked_index.append(j)
    if i in tracked_index:
        l = copy.copy(label)
        label[:, 0] = l[:, 1] #+ int(1080/2)
        label[:, 1] = l[:, 0] #- int(1920/2)
    return label


def face_center(label, face=1):
    if face == 1:
        points = [0, 1, 2, 3]
        line_1 = [0, 2]
        line_2 = [1, 3]
    elif face == 2:
        points = [0, 3, 4, 5]
        line_1 = [0, 4]
        line_2 = [3, 5]
    elif face == 3:
        points = [0, 5, 6, 1]
        line_1 = [0, 6]
        line_2 = [1, 5]
    
    # Extract points
    p = np.array([label[i,:] for i in points])
    q = np.array([label[i,:] for i in range(7)])
    # Calculate center point
    c = np.mean(p, axis=0)
    
    # Calculate intersection point
    intersection = find_intersection(q[line_1[0]], q[line_1[1]], q[line_2[0]], q[line_2[1]])
    
    # Check if c coincides with intersection point
    if np.linalg.norm(intersection - c)< np.linalg.norm(q[line_1[1]]-q[line_1[0]])*0.05:
        return True
    else:
        return False
    
def find_intersection(p1, p2, p3, p4):
    x1, y1 = p1[0], p1[1]
    x2, y2 = p2[0], p2[1]
    x3, y3 = p3[0], p3[1]
    x4, y4 = p4[0], p4[1]
    
    # Calculate slopes
    m1 = (y2 - y1) / (x2 - x1) if x2 != x1 else np.inf
    m2 = (y4 - y3) / (x4 - x3) if x4 != x3 else np.inf
    
    # Check if lines are parallel
    if np.isinf(m1) and np.isinf(m2):
        return np.nan, np.nan
    elif np.isinf(m1):
        x = x1
        y = m2 * (x - x3) + y3
    elif np.isinf(m2):
        x = x3
        y = m1 * (x - x1) + y1
    else:
        # Calculate intersection point
        x = (m1*x1 - m2*x3 + y3 - y1) / (m1 - m2)
        y = m1 * (x - x1) + y1
    
    return np.array([x, y])


def sort_label(label):
    # determine inner corner of the cube
    # this point is inside the convex hull
    hull = ConvexHull(label)
    hull_vertices = hull.vertices
    point_not_on_hull = None
    for i, point in enumerate(label):
        if i not in hull_vertices:
            point_not_on_hull = point
            break
    center = point_not_on_hull
    # Sort points clockwise relative to the center
    sorted_label = sorted(label, key=lambda p: -(np.arctan2(p[1] - center[1], p[0] - center[0]) - 2 * np.pi) % (2 * np.pi))
    if not face_center(np.array(sorted_label), face=1):
        s = [sorted_label[0]]
        for i in range(6):
            s.append(sorted_label[1:][(i-1)])
        sorted_label = s
    # take (x,y) coordinates and split them
    result = []
    for y in sorted_label:
        result.append(y[0])
        result.append(y[1])
    return np.array(result)

def reconstruct_label(label):
    import numpy as np
    return np.array([(label[i], label[i+1]) for i in range(0, len(label), 2)])


def hungarian_loss(y_true, y_pred):
    """
    Hungarian loss with TensorFlow graph compatibility.
    """
    batch_size = tf.shape(y_true)[0]
    num_points = y_true.shape[1] // 2  # For 7 points, 2 coordinates each
    
    y_true = tf.reshape(y_true, (batch_size, num_points, 2))
    y_pred = tf.reshape(y_pred, (batch_size, num_points, 2))

    total_loss = 0.0
    
    # Process each sample in the batch
    for i in tf.range(batch_size):
        # Get i-th sample's true/pred points
        true_points = y_true[i]
        pred_points = y_pred[i]
        
        # Compute pairwise distances
        dist_matrix = tf.norm(
            tf.expand_dims(true_points, 1) - tf.expand_dims(pred_points, 0), 
            axis=-1
        )
        
        # Convert to numpy array INSIDE a tf.py_function wrapper
        cost_matrix = tf.py_function(
            func=lambda x: x.numpy(), 
            inp=[dist_matrix], 
            Tout=tf.float32
        )
        
        # Find optimal assignment using Hungarian algorithm
        row_ind, col_ind = tf.numpy_function(
            func=lambda x: linear_sum_assignment(x),
            inp=[cost_matrix],
            Tout=(tf.int64, tf.int64)
        )
        
        # Compute MSE for matched points
        matched_true = tf.gather(true_points, row_ind)
        matched_pred = tf.gather(pred_points, col_ind)
        sample_loss = tf.reduce_mean(tf.square(matched_true - matched_pred))
        
        total_loss += sample_loss
    
    return total_loss / tf.cast(batch_size, tf.float32)