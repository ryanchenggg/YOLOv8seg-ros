import rospy
import message_filters
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
import random
import cv2
# Assume overlay contains your YOLO and overlay_seg function, please replace with the actual module if it's different
import overlay 

model = overlay.YOLO('./seg_0924.pt')
class_names = model.names
colors = [[random.randint(0, 255) for _ in range(3)] for _ in class_names]

class ImageProcessor:
    def __init__(self):
        self.bridge = CvBridge()
        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None
        self.color_sub = message_filters.Subscriber('/camera/color/image_raw', Image)
        self.depth_sub = message_filters.Subscriber('/camera/depth/image_rect_raw', Image)
        self.info_sub = rospy.Subscriber("/camera/depth/camera_info", CameraInfo, self.info_callback)
        self.ts = message_filters.ApproximateTimeSynchronizer([self.color_sub, self.depth_sub], queue_size=10, slop=0.1)
        self.ts.registerCallback(self.callback)
        # New image publisher for Rviz
        self.image_pub = rospy.Publisher("/processed_image", Image, queue_size=1)

    def info_callback(self, data):
        if not self.fx:
            self.fx = data.K[0]
            self.fy = data.K[4]
            self.cx = data.K[2]
            self.cy = data.K[5]

    def callback(self, color_msg, depth_msg):
        try:
            cv_color_image = self.bridge.imgmsg_to_cv2(color_msg, "bgr8")
            cv_depth_image = self.bridge.imgmsg_to_cv2(depth_msg, "16UC1")
        except CvBridgeError as e:
            print(e)
            return

        # 在此处加入您的图像处理代码
        results = model.predict(cv_color_image)  # segmentation using YOLO
        result = results[0]
        masks = result.masks
        mask1 = masks[0]
        mask = mask1.data[0].numpy()
        polygon = mask1.xy[0]
        mask_img = Image.fromarray(mask, "I")

        timestr = "%.6f" % color_msg.header.stamp.to_sec()
        image_name = timestr + ".png"

        # overlay_seg function handles the segmentation and overlay
        processed_img = overlay.overlay_seg(cv_color_image, image_name, cv_depth_image)  

        
        # Convert processed image back to Image msg and publish
        try:
            ros_image = self.bridge.cv2_to_imgmsg(processed_img, "bgr8")
        except CvBridgeError as e:
            print(e)
        
        self.image_pub.publish(ros_image)

if __name__ == '__main__':
    rospy.init_node('image_processor', anonymous=True)
    ip = ImageProcessor()
    rospy.spin()
