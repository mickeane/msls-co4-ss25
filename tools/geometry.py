from scipy.spatial import KDTree
from scipy.optimize import minimize
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv


def align_points(data_a, data_b, 
                 n_samples=None, 
                 n_attemps=10, 
                 with_scale=True,
                 method="SLSQP",
                 x_init=None):
    """
    Aligns two point sequences by minimizing the average distance
    between the points. The alignment is done by translating, 
    rotating and optionally scaling the point sequence b.

    IDEA: In case we get trapped in local minima, it may help to 
          sample x_init randomly (especially dx and dy). For now
          this is disabled.

    Args:
        data_a:     (n, 2) array of points
        data_b:     (m, 2) array of points
        n_samples:  Number of points to resample to. If None, no 
                    resampling is done. Affects the computation time
                    and precision of the result.
        n_attemps:  Number of times to run the optimization with
                    different initial parameters. The result with the
                    smallest loss is returned.
        with_scale: If True, the scale is optimized. If False, the
                    scale is fixed to 1.
        method:     Optimization method. See scipy.optimize.minimize
        x_init:     Initial parameters for the optimization.
                    Order: (dx, dy, phi, [scale])
                    The fourth parameter is optional. If it is not
                    provided, the scale is fixed to 1. If x_init 
                    is provided, n_attemps is ignored.

    Returns:
        dx, dy: Translation
        phi:    Rotation angle (in degrees)
        scale:  Scaling factor
    """
    def preprocess(data, scale, n_samples=None):
        """For the optimization to work well, the data should 
        be centered and scaled. Also, the number of data points
        can be standardized by resampling the point sequence."""
        if n_samples is not None:
            poly = interp1d(range(data.shape[0]), data, axis=0)
            data = poly(np.linspace(0, data.shape[0] - 1, n_samples))
            
        center = data.mean(axis=0)
        return (data - center) / scale, center, scale
    
    def transform(data, dx, dy, phi, scale):
        """Translate, rotate and optionally scale the data."""
        rot = np.array([[np.cos(phi), -np.sin(phi)], 
                        [np.sin(phi), np.cos(phi)]])
        data = data.dot(rot) + [dx, dy]
        if scale is not None:
            data *= scale
        return data
    
    def fun_(params, *args):
        dx, dy, phi = params[:3]
        # scale is an optional parameter
        scale = None if len(params) < 4 else params[3]
        data_b, kdtree_a, info = args
        return fun(data_b=data_b, kdtree_a=kdtree_a, 
                   dx=dx, dy=dy, phi=phi, scale=scale,
                   info=info)

    def fun(data_b, kdtree_a, dx, dy, phi, scale, info):
        data_bt = transform(data_b, dx, dy, phi, scale)
        dists, _ = kdtree_a.query(data_bt)
        dist_mean = dists.mean()
        if info is not None:
            info["dist_mean"] = dist_mean
            info["dist_max"] = dists.max()
        # Alternative: (smoothly) penalize negative scaling factors
        #   dist_mean - min(scale**2, 0)*100
        # Note: scale = -1 is equivalent to phi = phi + pi
        # TODO: Penalize very small scales!
        scale_penalty = 1/scale
        return dist_mean #+ scale_penalty**2
    
    def plot(data_a, data_b, params):
        dx, dy, phi, scale = params
        data_b = transform(data_b, dx, dy, phi, scale)
        plt.plot(data_a[:,0], data_a[:,1], 'r-')
        plt.plot(data_b[:,0], data_b[:,1], 'b-')
        plt.axis("square")
        plt.show()

    def print_result(params, loss):
        # Report results in a consistent way:
        #   - scale always positive
        #   - phi in [0, 2*pi)
        # Note: scale = -1 is equivalent to phi = phi + pi
        dx, dy, phi, scale = params
        if scale < 0:
            scale = -scale
            phi += np.pi
            phi = phi % (2*np.pi)
        print("Optimization results:")  
        print("    Loss: % .3f" % loss)
        print("    dx:   % .3f" % dx)
        print("    dy:   % .3f" % dy)
        print("    phi:  % .3f (%.1f°)" % (phi, phi/np.pi*180))
        if scale is not None:
            print("    scale:% .3f" % scale)
        else:
            print("    scale: fixed")

    def align_step(kdtree_a, data_b, x_init, tol=1e-6, method="CG"):
        """Perform the alignment for a particular initial guess x_init.
        """
        info = {}
        # Loss: mean distance between the points after transformation
        res = minimize(fun_, x0=x_init, args=(data_b, kdtree_a, info),
                       method=method, tol=tol)
        params = res.x if len(res.x) == 4 else (*res.x, None)
        return params, info
    
    scale_prep = data_a.std(axis=0).mean()
    data_a, center_a, scale_a = preprocess(data_a, scale_prep, n_samples)
    data_b, center_b, scale_b = preprocess(data_b, scale_prep, n_samples)

    # KDTree is a data structure for fast nearest neighbor search.
    kdtree_a = KDTree(data_a)

    if x_init is None and n_attemps >= 1:
        # Idea: To avoid local minima, we can try different initial
        #       parameters and pick the best result. Here, we sample 
        #       different angles and (optionally) different 
        #       translations.
        random_state = np.random.RandomState(42)
        use_random = False
        
        phis = np.linspace(0, 2*np.pi, n_attemps)
        scale = 1. if with_scale else None
        results = []
        losses = []
        print("Optimization progress:")
        for i, phi in enumerate(phis):
            if use_random:
                # Note: We have preprocessed the data space. Therefore,
                #       it is sufficient to sample dx and dy from the
                #       interval [-1, 1].
                dx = random_state.uniform(-1, 1)
                dy = random_state.uniform(-1, 1)
                x_init = [dx, dy, phi]
            else:
                x_init = [0, 0, phi]
            if with_scale:
                x_init.append(scale)
            
            params, info = align_step(kdtree_a, data_b, x_init, method=method)
            results.append(params)
            losses.append(info["dist_mean"])
            print("    i=%d: mean_dist = %4.1f, max_dist = %4.1f" 
                  % (i+1, info["dist_mean"]*scale_prep, info["dist_max"]*scale_prep))
        
        idx_opt = np.argmin(losses)
        params = results[idx_opt]
        loss = losses[idx_opt]
    else:
        params, loss = align_step(kdtree_a, data_b, x_init, method=method)
        print_result(params=params, loss=loss)
        plot(data_a, data_b, params=params)

    print()
    print_result(params=params, loss=loss)
    print()
    plot(data_a, data_b, params=params)
    
    # Compute the final transform:
    #  1. Shift center of point cloud B to the origin
    #  2. Apply the rotation
    #  3. Scale the points
    #  4. Shift the origin back to the center of the point cloud A

    # Transformation matrix
    dx, dy, phi, scale = params
    trafo = cv.getRotationMatrix2D(center_b, phi*180/np.pi, scale)
    trafo[0:2,2] += ([dx*scale_prep, dy*scale_prep] + (center_a-center_b))
    return trafo


###############################################################################
# Attempt using ICP...
###############################################################################
# import numpy as np
# from sklearn.neighbors import NearestNeighbors


# def best_fit_transform(A, B):
#     '''
#     Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions
#     Input:
#       A: Nxm numpy array of corresponding points
#       B: Nxm numpy array of corresponding points
#     Returns:
#       T: (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
#       R: mxm rotation matrix
#       t: mx1 translation vector
#     '''

#     assert A.shape == B.shape

#     # get number of dimensions
#     m = A.shape[1]

#     # translate points to their centroids
#     centroid_A = np.mean(A, axis=0)
#     centroid_B = np.mean(B, axis=0)
#     AA = A - centroid_A
#     BB = B - centroid_B

#     # rotation matrix
#     H = np.dot(AA.T, BB)
#     U, S, Vt = np.linalg.svd(H)
#     R = np.dot(Vt.T, U.T)

#     # special reflection case
#     if np.linalg.det(R) < 0:
#        Vt[m-1,:] *= -1
#        R = np.dot(Vt.T, U.T)

#     # translation
#     t = centroid_B.T - np.dot(R,centroid_A.T)

#     # homogeneous transformation
#     T = np.identity(m+1)
#     T[:m, :m] = R
#     T[:m, m] = t

#     return T, R, t


# def nearest_neighbor(src, dst):
#     '''
#     Find the nearest (Euclidean) neighbor in dst for each point in src
#     Input:
#         src: Nxm array of points
#         dst: Nxm array of points
#     Output:
#         distances: Euclidean distances of the nearest neighbor
#         indices: dst indices of the nearest neighbor
#     '''
#     assert src.shape == dst.shape

#     neigh = NearestNeighbors(n_neighbors=1)
#     neigh.fit(dst)
#     distances, indices = neigh.kneighbors(src, return_distance=True)
#     return distances.ravel(), indices.ravel()


# def icp(A, B, init_pose=None, max_iterations=20, tolerance=0.001):
#     '''
#     The Iterative Closest Point method: finds best-fit transform that maps points A on to points B
#     Input:
#         A: Nxm numpy array of source mD points
#         B: Nxm numpy array of destination mD point
#         init_pose: (m+1)x(m+1) homogeneous transformation
#         max_iterations: exit algorithm after max_iterations
#         tolerance: convergence criteria
#     Output:
#         T: final homogeneous transformation that maps A on to B
#         distances: Euclidean distances (errors) of the nearest neighbor
#         i: number of iterations to converge
#     '''

#     assert A.shape == B.shape

#     # get number of dimensions
#     m = A.shape[1]

#     # make points homogeneous, copy them to maintain the originals
#     src = np.ones((m+1,A.shape[0]))
#     dst = np.ones((m+1,B.shape[0]))
#     src[:m,:] = np.copy(A.T)
#     dst[:m,:] = np.copy(B.T)

#     # apply the initial pose estimation
#     if init_pose is not None:
#         src = np.dot(init_pose, src)

#     prev_error = 0

#     for i in range(max_iterations):
#         # find the nearest neighbors between the current source and destination points
#         ret = nearest_neighbor(src[:m,:].T, dst[:m,:].T)
#         distances, indices = ret

#         # compute the transformation between the current source and nearest destination points
#         T,_,_ = best_fit_transform(src[:m,:].T, dst[:m,indices].T)

#         # update the current source
#         src = np.dot(T, src)

#         # check error
#         mean_error = np.mean(distances)
#         if np.abs(prev_error - mean_error) < tolerance:
#             break
#         prev_error = mean_error

#     # calculate final transformation
#     T,_,_ = best_fit_transform(A, src[:m,:].T)

#     return T, distances, i


###############################################################################
# Attempt using Procrusted
###############################################################################
# from scipy.spatial import procrustes
# from scipy.interpolate import interp1d
# a = contourA[:,0,:]
# b = contourB[:,0,:]
# n = min(a.shape[0], b.shape[0])

# mean_a = np.mean(a, axis=0)
# mean_B = np.mean(b, axis=0)
# T_init = np.eye(3)
# T_init[:2, 2] =   mean_B - mean_a

# print(T_init)

# # Resample point sequence a to have exactly n points
# poly_a = interp1d(range(a.shape[0]), a, axis=0)
# resampled_a = poly_a(np.linspace(0, a.shape[0] - 1, n))

# # Resample point sequence b to have exactly n points
# poly_b = interp1d(range(b.shape[0]), b, axis=0)
# resampled_b = poly_b(np.linspace(0, b.shape[0] - 1, n))

# mtx1, mtx2, disparity = procrustes(resampled_a, resampled_b)

# T, distances, i = icp(resampled_a, resampled_b, T_init, max_iterations=3000)

# s = T[:2,:2].dot(resampled_a.T)
# sx = s[0] - T[0,2]
# sy = s[1] - T[1,2]

# fig, ax = plt.subplots()
# ax.plot(a[:,0], a[:,1], '-')
# ax.plot(b[:,0], b[:,1], '-')
# ax.axis("equal")

# fig, ax = plt.subplots()
# ax.plot(resampled_b[:,0], resampled_b[:,1], '-')
# ax.plot(sx, sy, '-')
# ax.axis("equal")
# plt.show()
