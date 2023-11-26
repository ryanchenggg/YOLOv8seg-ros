# coding:utf-8
#!/usr/bin/python
    
# Extract images from a bag file.
    
import roslib
import rosbag
import rospy
import cv2
import torch
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from cv_bridge import CvBridgeError
import math
    
rgb_path = '/home/ryan0309/bagfile/rgb/'   #已經建立好的儲存rgb彩色圖像的目錄
depth_path= '/home/ryan0309/bagfile/depth/' # 已經建立好的儲存深度圖像的目錄
model = torch.hub.load('ultralytics/yolov5', 'custom', path='0703_best.pt', force_reload=True)

class ImageCreator_close():
    def __init__(self):
        self.bridge = CvBridge()
        i = 1
        with rosbag.Bag('/home/ryan0309/bagfile/2023-07-18-20-00-31.bag', 'r') as bag:  
            last_rgb_stamp = None
            closest_depth_msg = None
            closest_diff = None
            for topic, msg, t in bag.read_messages():
                # 獲取相機內參
                if topic == "/camera/depth/camera_info" and i==1:    
                    camera_info_msg = msg  # type: CameraInfo
                    self.fx = camera_info_msg.K[0]
                    self.fy = camera_info_msg.K[4]
                    self.cx = camera_info_msg.K[2]
                    self.cy = camera_info_msg.K[5]
                    i+=1  

                # 彩色圖像的topic
                if topic == "/camera/color/image_raw":  
                    try:
                        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
                    except CvBridgeError as e:
                        print(e)
                    timestr = "%.6f" % msg.header.stamp.to_sec()
                    image_name = timestr + ".png"
                    cv_image = cv2.resize(cv_image, (848,480))
                    #cv2.imwrite(f'test{timestr}.png',cv_image)
                    results = model(cv_image) #detecting
                    
                    #rendering bounding box
                    rendered_img = results.render()[0]
                    
                    # 更新彩色圖像的時間戳
                    last_rgb_stamp = msg.header.stamp

                    # 如果有最接近的深度圖像，儲存它
                    if closest_depth_msg is not None:
                        try:
                            depth_image = self.bridge.imgmsg_to_cv2(closest_depth_msg, "16UC1")
                        except CvBridgeError as e:
                            print(e)

                        #抓出辨識框座標
                        processed_img = self.process_results(results, depth_image, rendered_img)
                        
                        timestr = "%.6f" % closest_depth_msg.header.stamp.to_sec()
                        image_name = timestr + ".png"
                        cv2.imwrite(depth_path + image_name, depth_image)
                        cv2.imwrite(rgb_path + image_name, processed_img) 
                        print(f"successfully cut {image_name}")
                        
                    # 重置最接近的深度圖像
                    closest_depth_msg = None
                    closest_diff = None

                # 深度圖像的topic
                elif topic == "/camera/depth/image_rect_raw":  
                    # 如果還沒有彩色圖像的時間戳，忽略這個深度圖像
                    if last_rgb_stamp is None:
                        continue

                    # 計算這個深度圖像的時間戳與上一個彩色圖像的時間戳的差距
                    diff = abs(msg.header.stamp.to_sec() - last_rgb_stamp.to_sec())

                    # 如果這個深度圖像更接近上一個彩色圖像的時間戳，或者這是第一個深度圖像，則儲存它
                    if closest_diff is None or diff < closest_diff:
                        closest_depth_msg = msg
                        closest_diff = diff
        bag.close()
    
    def process_results(self, results, depth_image, rendered_img):
        # Access the attributes of the results object
        boxes = results.xyxy[0]  # bounding box coordinates for the only image
        gap_boxes = []
        for *box, conf, cls in boxes:
            if results.names[int(cls)] == 'gap' and conf > 0.65:
                gap_boxes.append(box)

        for gap_box in gap_boxes:
            # 代入座標進行計算
            x1, y1, x2, y2 = map(int, gap_box)
            # Ensure that the indexes are within image bounds
            y1 = min(y1, depth_image.shape[0] - 1)
            x1 = min(x1, depth_image.shape[1] - 1)
            y2 = min(y2, depth_image.shape[0] - 1)
            x2 = min(x2, depth_image.shape[1] - 1)

            left_top = depth_image[y1, x1]
            right_top = depth_image[y1, x2]
            left_bottom = depth_image[y2, x1]
            right_bottom = depth_image[y2, x2]

            X1, Y1 = self.calculation(x1, y1, left_top)
            X2, Y2 = self.calculation(x2, y1, right_top)
            X3, Y3 = self.calculation(x1, y2, left_bottom)
            X4, Y4 = self.calculation(x2, y2, right_bottom)

            height = math.sqrt((X3 - X1)**2 + (Y3 - Y1)**2)/1000
            width = math.sqrt((X2 - X1)**2 + (Y2 - Y1)**2)/1000
            print(f"length: {height:.3f}m, width: {width:.3f}m")

            # Put the text on the image
            text = f"L: {height:.3f}m, W: {width:.3f}m"
            cv2.putText(rendered_img, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (13, 70, 255), 2)

        return rendered_img
    
    def calculation(self, x, y, depth):
        X = (x - self.cx) * depth / self.fx
        Y = (y - self.cy) * depth / self.fy
        return X, Y

def resize_same(rgb_image, depth_image):
    image = cv2.resize(rgb_image, (depth_image.shape[1],depth_image.shape[0]))
    return image

if __name__ == '__main__':
    try:
        image_creator = ImageCreator_close()
        
    except rospy.ROSInterruptException:
        pass