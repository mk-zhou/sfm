import rospy
from sensor_msgs.msg import PointCloud2, PointField
from visualization_msgs.msg import Marker, MarkerArray
import std_msgs.msg
from geometry_msgs.msg import Point
import numpy as np
import sensor_msgs.point_cloud2 as pc2
from sklearn.decomposition import PCA

def read_point_cloud_data(file_path):
    points = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            if not line.startswith('#'):
                data = line.strip().split(' ')
                point = (
                    float(data[1]),
                    float(data[2]),
                    float(data[3]),
                    rgb_to_uint32(int(data[4]), int(data[5]), int(data[6]))
                )
                points.append(point)
    return points


def rgb_to_uint32(r, g, b):
    return (int(r) << 16) | (int(g) << 8) | int(b)


# corners_3d[0] = center + min_extent[0] * basis_vectors[0] + min_extent[1] * basis_vectors[1] + min_extent[2] * basis_vectors[2]
# corners_3d[1] = center + min_extent[0] * basis_vectors[0] + min_extent[1] * basis_vectors[1] + max_extent[2] * basis_vectors[2]
# corners_3d[2] = center + max_extent[0] * basis_vectors[0] + min_extent[1] * basis_vectors[1] + max_extent[2] * basis_vectors[2]
# corners_3d[3] = center + max_extent[0] * basis_vectors[0] + min_extent[1] * basis_vectors[1] + min_extent[2] * basis_vectors[2]
# corners_3d[4] = center + min_extent[0] * basis_vectors[0] + max_extent[1] * basis_vectors[1] + min_extent[2] * basis_vectors[2]
# corners_3d[5] = center + min_extent[0] * basis_vectors[0] + max_extent[1] * basis_vectors[1] + max_extent[2] * basis_vectors[2]
# corners_3d[6] = center + max_extent[0] * basis_vectors[0] + max_extent[1] * basis_vectors[1] + max_extent[2] * basis_vectors[2]
# corners_3d[7] = center + max_extent[0] * basis_vectors[0] + max_extent[1] * basis_vectors[1] + min_extent[2] * basis_vectors[2]

def compute_min_bounding_box(points):
    color_groups = {}

    for point in points:
        x, y, z, rgb = point
        r = (rgb >> 16) & 0xFF
        g = (rgb >> 8) & 0xFF
        b = rgb & 0xFF

        if (r, g, b) not in [(255, 255, 255), (0, 0, 0), (0, 255, 0)]:
            color = (r, g, b)
            if color not in color_groups:
                color_groups[color] = []
            color_groups[color].append(point)

    oriented_boxes = {}

    for color, group in color_groups.items():
        group = np.array(group)
        pca = PCA(n_components=3)
        pca.fit(group[:, :3])
        center = pca.mean_
        basis_vectors = pca.components_

        # Calculate the extent along each basis vector
        min_extent = np.min(np.dot(group[:, :3] - center, basis_vectors.T), axis=0)
        max_extent = np.max(np.dot(group[:, :3] - center, basis_vectors.T), axis=0)
        extents = max_extent - min_extent

        # Construct the oriented bounding box corners
        corners_3d = np.empty((8, 3))

        corners_3d[0] = center + min_extent[0] * basis_vectors[0] + min_extent[1] * basis_vectors[1] + min_extent[2] * basis_vectors[2]
        corners_3d[1] = center + min_extent[0] * basis_vectors[0] + min_extent[1] * basis_vectors[1] + max_extent[2] * basis_vectors[2]
        corners_3d[2] = center + max_extent[0] * basis_vectors[0] + min_extent[1] * basis_vectors[1] + max_extent[2] * basis_vectors[2]
        corners_3d[3] = center + max_extent[0] * basis_vectors[0] + min_extent[1] * basis_vectors[1] + min_extent[2] * basis_vectors[2]
        corners_3d[4] = center + min_extent[0] * basis_vectors[0] + max_extent[1] * basis_vectors[1] + min_extent[2] * basis_vectors[2]
        corners_3d[5] = center + min_extent[0] * basis_vectors[0] + max_extent[1] * basis_vectors[1] + max_extent[2] * basis_vectors[2]
        corners_3d[6] = center + max_extent[0] * basis_vectors[0] + max_extent[1] * basis_vectors[1] + max_extent[2] * basis_vectors[2]
        corners_3d[7] = center + max_extent[0] * basis_vectors[0] + max_extent[1] * basis_vectors[1] + min_extent[2] * basis_vectors[2]

        oriented_boxes[color] = corners_3d

    return oriented_boxes



def publish_point_cloud(points):
    rospy.init_node('point_cloud_publisher', anonymous=True)
    pub_cloud = rospy.Publisher('point_cloud', PointCloud2, queue_size=10)
    pub_marker = rospy.Publisher('visualization_marker_array', MarkerArray, queue_size=10)
    rate = rospy.Rate(10)

    fields = [
        PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(name='rgb', offset=12, datatype=PointField.UINT32, count=1),
    ]
    header = std_msgs.msg.Header()
    header.stamp = rospy.Time.now()
    header.frame_id = 'base_link'

    while not rospy.is_shutdown():
        bounding_boxes = compute_min_bounding_box(points)

        marker_array = MarkerArray()

        for color, corners_3d in bounding_boxes.items():
            marker = Marker()
            marker.header = header
            marker.ns = 'bounding_boxes'
            marker.id = len(marker_array.markers)
            marker.type = Marker.LINE_LIST
            marker.action = Marker.ADD
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.01
            marker.color.r = color[0] / 255.0
            marker.color.g = color[1] / 255.0
            marker.color.b = color[2] / 255.0
            marker.color.a = 1.0

            lines = [[0, 1], [1, 2], [2, 3], [3, 0],
                     [4, 5], [5, 6], [6, 7], [7, 4],
                     [0, 4], [1, 5], [2, 6], [3, 7]]

            for line in lines:
                p1 = corners_3d[line[0]]
                p2 = corners_3d[line[1]]
                marker.points.append(Point(p1[0], p1[1], p1[2]))
                marker.points.append(Point(p2[0], p2[1], p2[2]))
                marker_array.markers.append(marker)

        pub_marker.publish(marker_array)

        cloud = pc2.create_cloud(header, fields, points)
        pub_cloud.publish(cloud)

        rate.sleep()


if __name__ == '__main__':
    file_path = 'points3D.txt'
    points = read_point_cloud_data(file_path)
    print("read_data is ok")
    publish_point_cloud(points)
