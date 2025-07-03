#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import cv2
import rosbag
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


def main():
    parser = argparse.ArgumentParser(description="Extract images from a ROS bag.")
    parser.add_argument("bag_file", help="Input ROS bag.")
    parser.add_argument("output_dir", help="Output directory.")
    parser.add_argument("image_topic", help="Image topic.")
    args = parser.parse_args()

    print("Extract images from {} on topic {} into {}".format(
        args.bag_file, args.image_topic, args.output_dir))

    bag = rosbag.Bag(args.bag_file, "r")
    bridge = CvBridge()

    os.makedirs(args.output_dir, exist_ok=True)


    # Use frame number to name the images
    # count = 0
    # for topic, msg, t in bag.read_messages(topics=[args.image_topic]):
    #     cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
    #     filename = os.path.join(args.output_dir, "frame{:06d}.png".format(count))
    #     cv2.imwrite(filename, cv_img)
    #     print("Wrote image {}".format(count))
    #     count += 1


    # Use Timestamp to name the images
    for topic, msg, t in bag.read_messages(topics=[args.image_topic]):
        cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")

        stamp = t.to_sec()  # float: seconds (6 decimal places)
        filename = os.path.join(args.output_dir, f"frame{stamp:.6f}.png")

        cv2.imwrite(filename, cv_img)
        print(f"Wrote image at time {stamp:.6f}")


    bag.close()


if __name__ == '__main__':
    main()
