from pynq.overlays.base import BaseOverlay
from pynq.lib.video import *
import cv2
import numpy as np

import time  # 引入time模块

def init(hdmi_out,videoIn):  
    frame_in_w = 640
    frame_in_h = 480
    # 显示器设置: 640*480 @ 60Hz
    Mode = VideoMode(frame_in_w,frame_in_h,24)   
    hdmi_out.configure(Mode,PIXEL_RGB)
    hdmi_out.start()
    print("显示器设置完成.....")
    # USB摄像头设置    
    videoIn.set(cv2.CAP_PROP_FRAME_WIDTH, frame_in_w);
    videoIn.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_in_h);
    print("capture device is open: " + str(videoIn.isOpened()))
    print("USB摄像头设置完成.....")

#旋转裁剪车牌
def getPad(img,box_2D):
    points = cv2.boxPoints(box_2D)
    points_int = points.astype(int)  
    w_pad = box_2D[1][0]
    h_pad = box_2D[1][1]
    if(w_pad > h_pad):
        dir = 1   #顺时针旋转
    else:
        w_pad,h_pad = h_pad, w_pad
        dir = 0   #逆时针旋转  
    left = min(points_int[0][0],points_int[1][0],points_int[2][0],points_int[3][0])
    right = max(points_int[0][0],points_int[1][0],points_int[2][0],points_int[3][0])
    up = min(points_int[0][1],points_int[1][1],points_int[2][1],points_int[3][1])
    down = max(points_int[0][1],points_int[1][1],points_int[2][1],points_int[3][1]) 
    
    #此处进行边缘图片判断
    if(left < 0):
        if(up < 0):
            img_new = img[0:down,0:right] 
        elif(down > 480 ):
            img_new = img[up:480,0:right] 
        else:
            img_new = img[up:down,0:right] 
     
    elif(right > 640):
        if(up < 0):
            img_new = img[0:down,left:640] 
        elif(down >480 ):
            img_new = img[up:480,0:640] 
        else:
            img_new = img[up:down,0:640] 
            
    elif(down > 480):
        img_new = img[up:480,left:right] 
    elif(up < 0):
        img_new = img[0:down,left:right]
    else:
        img_new = img[up:down,left:right]   
    
    #修正后的图片
    #img_new = img[up:down,left:right]   
    
    w_pic = right - left
    h_pic = down - up
    if(w_pad > w_pic):
        border = (w_pad - w_pic) // 2 
        img_border  = cv2.copyMakeBorder(img_new,0,0,border,border,cv2.BORDER_CONSTANT,value = 0)
    else:
        border = 0
        img_border = img_new
    w = w_pic + 2 * border
    h = h_pic
    
    center = ( w // 2 + 1, h // 2 + 1)
    if(dir == 1):
        M = cv2.getRotationMatrix2D(center,box_2D[2],1.0)      
    else:
        M = cv2.getRotationMatrix2D(center,90 + box_2D[2],1.0)  
        
    img_new1 = cv2.warpAffine(img_border,M,(w,h))   
    left_new = center[0] - w_pad // 2 + 1
    right_new = center[0] + w_pad // 2 -1
    up_new = center[1] - h_pad // 2 + 1
    down_new = center[1] + h_pad // 2 -1
 
    img_final = img_new1[up_new:down_new,left_new:right_new]   
    img_re = cv2.resize(img_final,(220,70),interpolation = cv2.INTER_CUBIC)  
    return img_re   
    
    
def detect_pai(img_c):
      
    ticks = time.time()     
    print ("处理图像前时间戳为:", ticks)
    
    img = img_c    
    lower_blue = np.array([95, 80, 76])
    upper_blue = np.array([124, 255, 255])
    print("进入函数完成.....")
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)                            #将BGR图像转化到HSV的颜色空间         
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)                  #获取蓝色的掩膜版
    kernel_mask = np.ones((10, 10), np.uint8)
    mask_close = cv2.morphologyEx(mask_blue, cv2.MORPH_CLOSE, kernel_mask)#对掩膜版进行形态学闭运算       
    gray_m = cv2.medianBlur(mask_close,7)                                 #将黑白图像进行中值滤波
    print("颜色提取完成.....")
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)                           #将BGR图像转化成灰度图像
    gray_m1 = cv2.medianBlur(gray,3)
    canny = cv2.Canny(gray_m1,200,300)
    print("边缘检测完成.....")
    kernel_o = np.ones((7, 5))        
    canny_peng = cv2.dilate(canny, kernel_o, iterations=1) #膨胀
    real = cv2.bitwise_and(gray_m,canny_peng)
    Matrix = np.ones((20, 20), np.uint8)     
    img_close = cv2.morphologyEx(real, cv2.MORPH_CLOSE, Matrix)      #形态学闭运算
    
    kernel_o = np.ones((5, 5))        
    img_close_pen = cv2.dilate(img_close, kernel_o, iterations=1) #膨胀
    
    
    print("开始获取轮廓完成.....")
    img4,contours, hierarchy = cv2.findContours(img_close_pen, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)#获取轮廓
    print("获取轮廓完成.....")
    
    img_xiao = np.zeros((480,640,3),dtype = np.uint8)
    flag = 0
    center = (200,200)
    w_and_h = (100,200)
    angle = -40.5
    counter = 0   
    if(len(contours) != 0):
        for i in range(len(contours)):
            approx = cv2.approxPolyDP(contours[i],5,True)                   #轮廓近似 
            box_2d = cv2.minAreaRect(approx)
            w = box_2d[1][0]
            h = box_2d[1][1]
            w_real = max(w,h)
            h_real = min(w,h)
            if(h_real != 0):
                w_h = w_real/h_real
                if (w_h > 2 and w_h <  6.5 and w_real >50 and h >18):                
                    points = cv2.boxPoints(box_2d)
                    points_int = points.astype(int)
                    pts = np.array(points_int, np.int32)
                    pts = pts.reshape((-1,1,2))
                    #cv2.polylines(img,[pts],True,(0,255,0),2)   
                    
                    counter = counter + 1
                    
                    
                    center = (int(box_2d[0][0]),int(box_2d[0][1]))
                    w_and_h = (int(box_2d[1][0]),int(box_2d[1][1]))
                    angle = box_2d[2]                       
                    img_xiao_c = getPad(img,(center,w_and_h,angle))
                    img_xiao[80*flag+10 : 80*(flag+1),210:430] = img_xiao_c
                    flag = flag + 1
                    
                    cv2.polylines(img,[pts],True,(0,255,0),2)  
                    
                                                        
            else:
                print('高度为0')
    if(counter == 0):
        no_pad = cv2.imread('no_pad.jpg')
        img_nopad = cv2.cvtColor(no_pad, cv2.COLOR_BGR2RGB)
        img_xiao = img_nopad
        
        print('没有检测到车牌')
    print("准备退出函数.....") 
    ticks = time.time()     
    print ("处理图像后时间戳为:", ticks)
    
    return counter,img,img_xiao    
    
    


def main():
    base = BaseOverlay("base.bit")
    print("载入overlay成功.....")
    hdmi_out = base.video.hdmi_out
    videoIn = cv2.VideoCapture(0)
    init(hdmi_out,videoIn)
    
    
    ret,frame = videoIn.read()
    print("获取前两帧图像并舍弃.....")
    disp_mode = 0
    
    
    
    while videoIn.isOpened():
        ret,frame = videoIn.read()
        if (ret):
            img = frame      

            if (base.buttons[3].read()==1 and disp_mode == 0):
                disp_mode = 1
            elif(base.buttons[3].read()==1 and disp_mode == 1):
                disp_mode = 0
            
            
            if(disp_mode == 0):
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                outframe = hdmi_out.newframe()
                outframe[:] = rgb
                hdmi_out.writeframe(outframe)
            else:           
                counter,img_ig,img2 = detect_pai(img)   
                
                if(counter == 0):
                    outframe = hdmi_out.newframe()
                    outframe[:] = img2
                    hdmi_out.writeframe(outframe)
                    while True:
                        if (base.buttons[1].read()==1):
                            disp_mode = 0
                            break
                else:
                    rgb = cv2.cvtColor(img_ig, cv2.COLOR_BGR2RGB)
                    outframe = hdmi_out.newframe()
                    outframe[:] = rgb
                    hdmi_out.writeframe(outframe)                   
                    while True:
                        if (base.buttons[2].read()==1):
                            break
                    rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
                    outframe = hdmi_out.newframe()
                    outframe[:] = rgb
                    hdmi_out.writeframe(outframe)                                     
                    while True:
                        if (base.buttons[1].read()==1):
                            disp_mode = 0
                            break    
            
        else:
            raise RuntimeError("Error while reading from camera.")

        if (base.buttons[0].read()==1):
            break
            
    print("测试结束.....") 
    videoIn.release()        
    hdmi_out.stop()
    del hdmi_out
     
    print('this message is from main function')

 

if __name__ == '__main__':
    main()

