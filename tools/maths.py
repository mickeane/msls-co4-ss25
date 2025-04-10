import numpy as np

def intersect_lines_parametric(m1, q1, m2, q2):
    """
    Returns the intersection point of two lines given in parametric form.
    
    Arguments:
        m1:     Slope of line 1.
        q1:     Intercept of line 1.
        m2:     Slope of line 2.
        q2:     Intercept of line 2.
        
    Returns:
        x:      Intersection point.
        
    Examples:
        intersect_lines_parametric(1, 0, -1, 1)
        intersect_lines_parametric(1, 0, 1, 0)
        intersect_lines_parametric(1, 0, 1, 1)
        intersect_lines_parametric(1, 0, 0, 1)
    """
    if m1 == m2:
        return None
    x = (q2 - q1) / (m1 - m2)
    y = m1 * x + q1
    return [x, y]


def intersect_segments(p1, p2, q1, q2, extend=False):
    """
    Returns the intersection point of two line segments.
    
    Arguments:
        p1:     First point of line 1.
        p2:     Second point of line 1.
        q1:     First point of line 2.
        q2:     Second point of line 2.
        extend: Intersect outside the line segments.
        
    Returns:
        x:      Intersection point.
        
    Examples:
        intersect_lines([0,0], [1,1], [0,1], [1,0])
        intersect_lines([0,0], [1,1], [0,1], [0,2])
        intersect_lines([0,0], [1,1], [0,1], [1,1])
        intersect_lines([0,0], [1,1], [1,0], [2,1])
    """
    xdiff = (p1[0] - p2[0], q1[0] - q2[0])
    ydiff = (p1[1] - p2[1], q1[1] - q2[1])
    div = np.linalg.det([xdiff, ydiff])
    if div == 0:
        return None
    d = (np.linalg.det([p1, p2]), np.linalg.det([q1, q2]))
    x = np.linalg.det([d, xdiff]) / div
    y = np.linalg.det([d, ydiff]) / div
    
    # Check if intersection is within the line segments
    if extend:
        return [x, y]
    if (min(p1[0], p2[0]) <= x <= max(p1[0], p2[0]) and
        min(q1[0], q2[0]) <= x <= max(q1[0], q2[0]) and
        min(p1[1], p2[1]) <= y <= max(p1[1], p2[1]) and
        min(q1[1], q2[1]) <= y <= max(q1[1], q2[1])):
        return [x, y]
    else:
        return None
    
    
def intersect_line_with_segment(m, q, p1, p2, extend=False):
    """
    Returns the intersection point of a line and a line segment.
    
    Arguments:
        m:      Slope of the line.
        q:      Intercept of the line.
        p1:     First point of the line segment.
        p2:     Second point of the line segment.
        extend: Intersect outside the line segment.
        
    Returns:
        x:      Intersection point.
        
    Examples:
        intersect_line_with_segment(1, 0, [0,10], [4,0])
        intersect_line_with_segment(1, 0, [0,10], [2,4])
        intersect_line_with_segment(1, 0, [0,10], [2,4], extend=True)
    """
    m1, q1 = m, q
    m2 = (p1[1] - p2[1]) / (p1[0] - p2[0])
    q2 = p1[1] - m2 * p1[0]
    
    x = intersect_lines_parametric(m1, q1, m2, q2)
    if extend:
        return x
    elif (min(p1[0], p2[0]) <= x[0] <= max(p1[0], p2[0]) and
          min(p1[1], p2[1]) <= x[1] <= max(p1[1], p2[1])):
        return x
    else:
        return None
        
    