#from IPython.display import HTML
import numpy as np
import cv2
import pickle
import glob 
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
#%matplotlib inline

def calibrate(nx=6,ny=9,calib_img_path='./camera_cal/calibration*.jpg'):
    
    #prepare object points , like (0,0,0), (1,0,0), (2,0,0) ... (8,5,0)
    objp = np.zeros((nx*ny,3), np.float32)
    objp[:,:2] = np.mgrid[0:ny,0:nx].T.reshape(-1,2)

    #Arrays to store object points and image points from all the images
    objpoints = [] # 3d points to real world space
    imgpoints = [] # 2d points in image plane

    #Make a list of calibration images
    images = glob.glob(calib_img_path)
    
    for indx, fname in enumerate(images):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        #find chessboard corners
        ret, corners = cv2.findChessboardCorners(gray,(9,6))

        #if found, add object points, image points
        if ret == True:
            print("working on",fname)
            objpoints.append(objp)
            imgpoints.append(corners)

            #draw and display the coreners
            cv2.drawChessboardCorners(img,(9,6),corners,ret)
            write_name = './After_corner_detection/'+'corners_found'+str(indx)+'.jpg'
            cv2.imwrite(write_name,img)

    img = cv2.imread('./camera_cal/calibration2.jpg')

    img_size = (img.shape[1],img.shape[0])

    #On camera calibration given object points and image points
    retval,cameraMatx,distCoeff,rvecs,tvecs = cv2.calibrateCamera(objpoints,imgpoints,img_size,None,None)

    #save the camera calibration values for later use
    dist_pickle = {}
    dist_pickle["mtx"] = cameraMatx
    dist_pickle["dst"] = distCoeff
    dist_pickle["rvecs"] = rvecs
    dist_pickle["tvecs"] = tvecs
    pickle.dump(dist_pickle,open("./calibration.p","wb"))

    #Read  in the saved objpoints and imgpoints
    dist_pickle = pickle.load(open("./calibration.p","rb"))
    mtx = dist_pickle["mtx"]
    dst = dist_pickle["dst"]
    rvecs = dist_pickle["rvecs"]
    tvecs = dist_pickle["tvecs"]
    
    return mtx,dst


def undistort(image,mtx,dist, debug=False):
    un_img = cv2.undistort(image,mtx,dist,None,mtx)
    if debug==True:
        # Visualize undistortion
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
        ax1.imshow(image)
        ax1.set_title('Original Image', fontsize=30)
        ax2.imshow(un_img)
        ax2.set_title('Undistorted Image', fontsize=30)
    return un_img

def abs_threshold_sobel(image,orient='x',thresh = (0,255)):
    gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    #Thresholding in 'x' axis
    if (orient=='x'):
        sobel = cv2.Sobel(gray,cv2.CV_64F,1,0)
    #Thresholding in y axis
    elif(orient=='y'):
        sobel = cv2.Sobel(gray,cv2.CV_64F,0,1)
    #Calculate absolute of thresholds
    abs_sobel = np.absolute(sobel)
    #Scale Thresholds in range 0 - 1
    scaled_sobel = np.uint8(255.*abs_sobel/np.max(abs_sobel))
    binary_img = np.zeros_like(scaled_sobel)
    binary_img[(sobel >= thresh[0])&(sobel <= thresh[1])]=1

    return binary_img

def color_threshold(image,s_thresh = (0,255),v_thresh = (0,255)):
    #Convert image from RGB to HLS and threshold for s channel
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0])&(s_channel <= s_thresh[1])] =1
    
    #Convert image from RGB to HSV and threshold for v channe;
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    v_channel = hsv[:,:,2]
    v_binary = np.zeros_like(v_channel)
    v_binary[(v_channel >= v_thresh[0])&(v_channel <= v_thresh[1])] =1
    
    #Combine both the thresholds
    output = np.zeros_like(s_channel)
    output[(s_binary ==1)&(v_binary ==1)] =1

    return output

def mag_thresh(image, sobel_kernel=3, mag_thresh=(0,255)):
    #Gradient magnitude thresholding using both x and y axis
    gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=sobel_kernel)
    sobely = cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=sobel_kernel)
    gradmag = np.sqrt((sobelx**2)+(sobely**2))
    scale_factor = np.max(gradmag)/255
    gradmag = (gradmag/scale_factor).astype(np.uint8)
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag>=mag_thresh[0])&(gradmag<=mag_thresh[1])] =1
    
    return binary_output

def dir_threshold(image,sobel_kernel=3,thresh=(0,np.pi/2)):
    gray= cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    sobelx= cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=sobel_kernel)
    sobely= cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=sobel_kernel)
    with np.errstate(divide='ignore',invalid='ignore'):
        absgradir = np.absolute(np.arctan(sobely/sobelx))
        binary_output = np.zeros_like(absgradir)
        binary_output[(absgradir >= thresh[0])&(absgradir <= thresh[1])]=1
    return binary_output

def perspective_transform(image, debug=False, size_top=70, size_bottom=370, offset_top=0, offset_bot=0):
    height, width = image.shape[0:2]
    output_size = height/2

    src = np.float32([[(width/2) - size_top, height*0.65], [(width/2) + size_top, height*0.65], [(width/2) + size_bottom, height-50], [(width/2) - size_bottom, height-50]])
    dst = np.float32([[(width/2) - output_size, (height/2) - output_size], [(width/2) + output_size, (height/2) - output_size], [(width/2) + output_size, (height/2) + output_size], [(width/2) - output_size, (height/2) + output_size]])
    
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(image, M, (width, height), flags=cv2.INTER_LINEAR)
    Minv = cv2.getPerspectiveTransform(dst, src)
           
    return warped, M, Minv

def get_perspective_rectangles(image):
    size_top=70
    size_bottom=370
    height, width = image.shape[0:2]
    output_size = height/2

    src = np.float32([[(width/2) - size_top, height*0.65], [(width/2) + size_top, height*0.65], [(width/2) + size_bottom, height-50], [(width/2) - size_bottom, height-50]])
    dst = np.float32([[(width/2) - output_size, (height/2) - output_size], [(width/2) + output_size, (height/2) - output_size], [(width/2) + output_size, (height/2) + output_size], [(width/2) - output_size, (height/2) + output_size]])

    return src, dst

def Process_data_mag_abs_color(image_1,sobel_kernel=21,idx=0,
                 mag_thresh_0=20, mag_thresh_1=255, 
                 x_thresh_0 = 212, x_thresh_1 = 255,
                 y_thresh_0 = 185, y_thresh_1 = 255,
                 s_thresh_0=100,s_thresh_1=255,
                 v_thresh_0=70,v_thresh_1=255, debug=False):
    
    preprocess_img = np.zeros_like(image_1[:,:,2])
    gradx = abs_threshold_sobel(image_1, orient = 'x', thresh = (x_thresh_0,x_thresh_1))
    grady = abs_threshold_sobel(image_1, orient = 'y', thresh = (y_thresh_0,y_thresh_1))
    gradxy = mag_thresh(image_1,sobel_kernel,mag_thresh=(mag_thresh_0,mag_thresh_1))
    c_binary = color_threshold(image_1,s_thresh=(s_thresh_0,s_thresh_1),v_thresh=(v_thresh_0,v_thresh_1))
    dir_binary = dir_threshold(image_1, sobel_kernel, thresh=(0, np.pi/2))
    preprocess_img[(gradx==1)&(grady==1)|(gradxy==1)&(dir_binary==1)|(c_binary == 1)]=1
    #preprocess_img[(gradx==1)&(grady==1)] = 1
    #preprocess_img = cv2.bitwise_not(preprocess_img)
    preprocess_img = cv2.GaussianBlur(preprocess_img,(5,5),0)
    """
    (thresh,im_bw) = cv2.threshold(preprocess_img,128,255,cv2.THRESH_BINARY| cv2.THRESH_OTSU)
    (thresh,gradx) = cv2.threshold(gradx,128,255,cv2.THRESH_BINARY| cv2.THRESH_OTSU)
    (thresh,grady) = cv2.threshold(grady,128,255,cv2.THRESH_BINARY| cv2.THRESH_OTSU)
    (thresh,gradxy) = cv2.threshold(gradxy,128,255,cv2.THRESH_BINARY| cv2.THRESH_OTSU)
    """
    if debug==True:
        print("dir_binary",dir_binary.shape)
        print("gradxy_size",gradxy.shape)
        print("C_binary",c_binary.shape)
        print("grady_size",grady.shape)
        print("gradx_size",gradx.shape)
        #write_name = './test_images/invert/thresholded'+str(idx)+'.jpg'
        #cv2.imwrite(write_name,im_bw)
        #write_name = './test_images/invert/processed_gradx'+str(idx)+'.jpg'
        #cv2.imwrite(write_name,gradx )
        #write_name = './test_images/invert/processed_grady'+str(idx)+'.jpg'
        #cv2.imwrite(write_name,grady)
        #write_name = './test_images/invert/processed_c_binary'+str(idx)+'.jpg'

        #gradx=np.array(gradx)
        #imgplot = plt.imshow(c_binary,cmap='gray')
        #imgplot = plt.imshow(grady)

        #preprocess_img = cv2.bitwise_not(preprocess_img)
        #imgplot = plt.imshow(preprocess_img,cmap='Greys')
        #imgplot = plt.imshow(preprocess_img)
        #plt.savefig(write_name)
        #plt.imsave(write_name,preprocess_img,cmap='Greys')
    return preprocess_img

def polynomial_fit_1(binary_warped):
        # Assuming you have created a warped binary image called "binary_warped"
    # Take a histogram of the bottom half of the image
    # Assuming you have created a warped binary image called "binary_warped"
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]/2:,:], axis=0)

    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255

    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 9

    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)

    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Set the width of the windows +/- margin
    margin = 100

    # Set minimum number of pixels found to recenter window
    minpix = 50

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 

        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    
    #result = visualization(out_img,nonzeroy,nonzerox,left_lane_inds,right_lane_inds)
    
    result = out_img
    #plt.imshow(out_img)
    #plt.plot(left_fitx, ploty, color='yellow')
    #plt.plot(right_fitx, ploty, color='yellow')
    #plt.xlim(0, 1280)
    #plt.ylim(720, 0)
    return result, left_fitx, right_fitx, left_fit, right_fit, ploty, nonzeroy,nonzerox,left_lane_inds,right_lane_inds

def polynomial_fit_2(binary_warped,left_fit,right_fit):
    # Assume you now have a new warped binary image 
    # from the next frame of video (also called "binary_warped")
    # It's now much easier to find line pixels!
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin))) 
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))  

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    
    window_img = np.zeros_like(out_img)
    # Color in left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
    
    #result = visualization(result,nonzeroy,nonzerox,left_lane_inds,right_lane_inds)
    #plt.imshow(result)
    #plt.plot(left_fitx, ploty, color='yellow')
    #plt.plot(right_fitx, ploty, color='yellow')
    #plt.xlim(0, 1280)
    #plt.ylim(720, 0)
    return result, left_fitx, right_fitx, left_fit, right_fit, ploty,nonzeroy,nonzerox,left_lane_inds,right_lane_inds

def render_lane_detected(image,binary_img,ploty,left_fitx,right_fitx, Minv):
       
    """  
    #src, dst = get_perspective_rectangles(image)
    #Minv = cv2.getPerspectiveTransform(dst, src)
    
    warp_zero = np.zeros_like(image[:,:,0]).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0])) 
    # Combine the result with the original image
    result = cv2.addWeighted(image, 1.0, newwarp, 0.3, 0)
    res_img = cv2.resize(binary_img,(64,64),interpolation=cv2.INTER_AREA)
    #result = cv2.addWeighted(result,1.0,binary_img,0.5,0)
    #result = cv2.putText(result, "Bird,s view", (900, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, [0, 0, 0], 2)
    #plt.imshow(result)
    
    #lanes = visualization(result,nonzeroy)
    """
    
    src, dst = get_perspective_rectangles(image)
    Minv = cv2.getPerspectiveTransform(dst, src)
    
    warp_zero = np.zeros_like(image[:,:,0]).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0])) 
    # Combine the result with the original image
    #print(newwarp.shape)
    result = cv2.addWeighted(image, 1.0, newwarp, 0.3, 0)
    #print(result.shape)
    #res_img = cv2.resize(binary_img,(64,64),interpolation=cv2.INTER_AREA)
    #print(res_img.shape)
    #res_img = cv2.addWeighted(result,2.0,binary_img,0.5,0)
    #res_img = cv2.
    #result = cv2.addWeighted(result,1.0,res_img,1.0,0)
    #result = cv2.putText(result, "Bird,s view", (900, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, [0, 0, 0], 2)
    
    #lanes = visualization(result,nonzeroy)
    
    return result

def get_curvature(ploty, left_fitx, right_fitx):
    y_eval = np.max(ploty)

    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 20 / 720 # meters per pixel in y dimension
    xm_per_pix = 3.7 / 700 # meters per pixel in x dimension

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty*ym_per_pix, left_fitx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, right_fitx*xm_per_pix, 2)

    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

    # Now our radius of curvature is in meters
#     print(left_curverad, 'm', right_curverad, 'm')

    return (left_curverad+right_curverad)/2



def process_image(input_image,mtx, dst,prev_left_fit=0,prev_right_fit=0, video=False):
    
    # step 1: undistort image
    image_undistort = undistort(input_image,mtx,dst)
    
    # step 2: perspective transform
    image_transformed, M, Minv= perspective_transform(image_undistort)
    
    # step 3: detect binary lane markings
    image_binary = Process_data_mag_abs_color(image_transformed)
    
    # step 4: fit polynomials
    if (prev_left_fit is not None) and  (prev_right_fit is not None):
        out_img2, left_fitx, right_fitx, left_fit, right_fit, ploty,nonzeroy,nonzerox,left_lane_inds,right_lane_inds = polynomial_fit_2(image_binary, prev_left_fit, prev_right_fit)
        
        #print("test2")
    else:
        out_img2, left_fitx, right_fitx, left_fit, right_fit, ploty,nonzeroy,nonzerox,left_lane_inds,right_lane_inds = polynomial_fit_1(image_binary)
        #print("test1")
        #print(global_left_fit)
    
    # step 5: draw lane
    output_lane = render_lane_detected(input_image, image_transformed,ploty, left_fitx, right_fitx,Minv)
    
    
    # step 6: print curvature
    curv = get_curvature(ploty, left_fitx, right_fitx)
    output_curvature = cv2.putText(output_lane, "CURVATURE: " + str(int(curv)) + "m", (900, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, [0, 0, 0], 2)

    # step 7: print road position
    xm_per_pix = 3.7/700
    left_lane_pos = left_fitx[len(left_fitx)-1]
    right_lane_pos = right_fitx[len(right_fitx)-1]
    road_pos = (((left_lane_pos + right_lane_pos) / 2) - 640) * xm_per_pix
    output_road_pos = cv2.putText(output_lane, "OFFSET: {0:.2f}m".format(road_pos), (900, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, [0, 0, 0], 2)

    # output from processing step
    output_image = output_road_pos
        
    #print(nonzerox.shape)
    #print(nonzeroy.shape)
    #print(left_lane_inds.shape)
    #print(right_lane_inds.shape)
    #lanes_marked = visualization(processed,nonzeroy,nonzerox,left_lane_inds,right_lane_inds)
    
    #image_binary = np.stack[image_binary,image_binary,image_binary]
    #image_binary = np.dstack((image_binary, image_binary, image_binary))*255
    #print(image_undistort.shape)
    #print(image_transformed.shape)
    #print(image_binary.shape)
    #print(processed.shape)
    output_image = np.concatenate((image_undistort,output_image), axis=1)
    #vis2 = np.concatenate((image_binary,processed), axis=1)
    #print(vis1.shape)
    #print(vis2.shape)

    #vis3 = np.concatenate((vis1,image_binary), axis=0)
  
    #ax1.imshow(vis1)
    #ax1.set_title('vis', fontsize=30)
    #ax2.imshow(vis3)
    #ax2.set_title('vis', fontsize=30)
    
    # function should always output color images
    if len(output_image.shape) == 2:
        output_image =  cv2.cvtColor(np.float32(output_lane), cv2.COLOR_GRAY2RGB)
    else:
        return output_image , left_fit, right_fit

