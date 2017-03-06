from moviepy.editor import VideoFileClip
from LaneTracker import process_image, calibrate

g_prev_left_fit = None
g_prev_right_fit =None

def process_Video(input_image):
    global g_prev_left_fit
    global g_prev_right_fit
    output_image , left_fit, right_fit = process_image(input_image,g_mtx,g_dst,g_prev_left_fit,g_prev_right_fit)
    g_prev_left_fit = left_fit
    g_prev_right_fit = right_fit
    return output_image

if __name__ == '__main__':
    
    
    g_mtx,g_dst = calibrate(nx=6,ny=9,calib_img_path='./camera_cal/calibration*.jpg')
    project_output_file = "project_output_final.mp4"
    project_clip = VideoFileClip("project_video.mp4")
   
    project_output = project_clip.fl_image(process_Video)
    project_output.write_videofile(project_output_file, audio=False)