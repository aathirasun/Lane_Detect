import cv2
import numpy as np
import matplotlib.pyplot as plt
def canny(image):
    gr=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    blur=cv2.GaussianBlur(gr ,(5,5),0)
    canny=cv2.Canny(blur, 50,150)
    return canny
def Reg_of_interest(image):
    triangle=np.array([[200,700],[1100,700],[550,250]],np.int32)
    mask=np.zeros_like(image)
    dr=cv2.fillPoly(mask ,[triangle],(255,255,255))
    return dr
def displayLines(image,lines):
    line_img=np.zeros_like(img)
    if lines is not None:
        for line in lines:
            x1,y1,x2,y2=line.reshape(4)
            #print(x1,y1,x2,y2)
            cv2.line(line_img, (x1,y1) ,(x2,y2),(255,0,0),10)
    return line_img
def avg_slope(image,lines):
    left_fit=[] 
    right_fit=[]
    for line in lines:
        x1,y1,x2,y2=line.reshape(4)
        parameters=np.polyfit((x1,x2),(y1,y2),1)
        slope=parameters[0]
        intercept=parameters[1]
        if slope<0:
            left_fit.append((slope,intercept))
        else:
            right_fit.append((slope,intercept))
    left_fit_avg=np.average(left_fit, axis=0)
    right_fit_avg=np.average(right_fit, axis=0)
    #print("left",left_fit_avg)
    #print("right",right_fit_avg)
    left_line=make_coordinates(image,left_fit_avg)
    right_line=make_coordinates(image,right_fit_avg)
    return np.array([left_line,right_line])

def make_coordinates(image, line_parameters):
    slope,intercept=line_parameters
    #print(image.shape)
    y1=image.shape[0]
    y2=int(y1*(3/5))
    x1=int((y1-intercept)/slope)
    x2=int((y2-intercept)/slope)
    return np.array([x1,y1,x2,y2])


img=cv2.imread("test_image.jpg")
lane_img=np.copy(img)
canny_img =canny(lane_img)
draw=Reg_of_interest(canny_img)
masked_img=cv2.bitwise_and(canny_img,draw)
lines=cv2.HoughLinesP(masked_img,2,np.pi/180,100,np.array([]), minLineLength=40, maxLineGap=5)
line_img=displayLines(lane_img,lines)
#blend_img=cv2.bitwise_or(line_img,lane_img)
blend_img=cv2.addWeighted(lane_img,0.8,line_img,1,1)
avg_lines=avg_slope(lane_img,lines)
line_img=displayLines(lane_img,avg_lines)
blend_img=cv2.addWeighted(lane_img,0.8,line_img,1,1)
cv2.imshow("image",blend_img)
cv2.waitKey(0)
cv2.destroyAllWindows()    
