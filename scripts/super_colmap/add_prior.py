import sqlite3
import argparse
import re
# 连接到数据库
from os.path import join, isfile


def update_snapshot_id(project_path):
    # 连接到数据库
    database_path = join(project_path, 'database.db')
    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()
    timestamp = {}
    index_txt_path = join(project_path, 'index.txt')
    with open (index_txt_path, 'r') as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        line = line.strip().split(' ')
        snapshot_id = line[0]
        for j in range(2, len(line)):
            timestamp[line[j]] = snapshot_id
    try:
        # 检查是否存在snapshot_id列
        cursor.execute("PRAGMA table_info(images)")
        columns = cursor.fetchall()
        has_snapshot_id_column = any(column[1] == 'snapshot_id' for column in columns)

        if not has_snapshot_id_column:
            # 添加snapshot_id列
            cursor.execute("ALTER TABLE images ADD COLUMN snapshot_id INTEGER")

        # 查询所有的name
        cursor.execute("SELECT name FROM images")
        rows = cursor.fetchall()

        for row in rows:
            name = row[0]
            # 使用正则表达式提取name中的数字
            snapshot_id = timestamp[name]
            cursor.execute("UPDATE images SET snapshot_id = ? WHERE name = ?", (snapshot_id, name))

        # 提交更改
        conn.commit()
        print("数据更新成功！")

    except sqlite3.Error as error:
        print("数据更新失败:", error)

    finally:
        # 关闭数据库连接
        cursor.close()
        conn.close()


def add_images_prior(project_path):
    database_path = join(project_path, 'database.db')
    images_txt_path = join(project_path, 'images.txt')
    photo_info_dict = {}
    # qw,qx,qy,qz,tx,ty,tz
    # 打开文本文件
    with open(images_txt_path, 'r') as file:
        image_poses = file.readlines()[::2]

    # 遍历文件行
    for image_pose in image_poses:
        data = image_pose.strip().split()
        # 解析字段
        name = data[-1]
        photo_info_dict[name] = {
            'camera_id': int(data[8]),
            'prior_qw': float(data[1]),
            'prior_qx': float(data[2]),
            'prior_qy': float(data[3]),
            'prior_qz': float(data[4]),
            'prior_tx': float(data[5]),
            'prior_ty': float(data[6]),
            'prior_tz': float(data[7])
        }
    #print(photo_info_dict)
    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()
    query = "SELECT  name FROM images"
    cursor.execute(query)
    results = cursor.fetchall()
    for data in photo_info_dict.items():
        query = "UPDATE images SET camera_id = ?,prior_qw = ?, prior_qx = ?, prior_qy = ?, prior_qz = ?, prior_tx = ?, prior_ty = ?, prior_tz = ? WHERE name = ?"
        cursor.execute(query, (
            data[1]['camera_id'], data[1]['prior_qw'], data[1]['prior_qx'], data[1]['prior_qy'], data[1]['prior_qz'],
            data[1]['prior_tx'], data[1]['prior_ty'], data[1]['prior_tz'], data[0]))
    conn.commit()

    # 关闭数据库连接
    conn.close()


def add_cameras_prior(project_path):
    database_path = join(project_path, 'database.db')
    cameras_txt_path = join(project_path, 'cam_pose.txt')
    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()
    cursor.execute('''ALTER TABLE cameras ADD COLUMN qw FLOAT''')
    cursor.execute('''ALTER TABLE cameras ADD COLUMN qx FLOAT''')
    cursor.execute('''ALTER TABLE cameras ADD COLUMN qy FLOAT''')
    cursor.execute('''ALTER TABLE cameras ADD COLUMN qz FLOAT''')
    cursor.execute('''ALTER TABLE cameras ADD COLUMN tx FLOAT''')
    cursor.execute('''ALTER TABLE cameras ADD COLUMN ty FLOAT''')
    cursor.execute('''ALTER TABLE cameras ADD COLUMN tz FLOAT''')
    # 提交更改并关闭连接
    conn.commit()
    conn.close()

    photo_info_dict = {}
    # qw,qx,qy,qz,tx,ty,tz
    # 打开文本文件
    with open(cameras_txt_path, 'r') as file:
        image_poses = file.readlines()[::2]

    # 遍历文件行
    for image_pose in image_poses:
        data = image_pose.strip().split()
        # 解析字段
        name = int(data[0])
        photo_info_dict[name] = {
            'prior_qw': float(data[1]),
            'prior_qx': float(data[2]),
            'prior_qy': float(data[3]),
            'prior_qz': float(data[4]),
            'prior_tx': float(data[5]),
            'prior_ty': float(data[6]),
            'prior_tz': float(data[7])
        }

    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()
    query = "SELECT  camera_id FROM cameras"
    cursor.execute(query)
    results = cursor.fetchall()
    # print(results)
    for data in photo_info_dict.items():
        #print(data)
        #print(data[0])
        #print(data[1]['prior_qw'])
        query = "UPDATE cameras SET qw = ?, qx = ?, qy = ?, qz = ?, tx = ?, ty = ?, tz = ? WHERE camera_id = ?"
        cursor.execute(query, (
            data[1]['prior_qw'], data[1]['prior_qx'], data[1]['prior_qy'], data[1]['prior_qz'], data[1]['prior_tx'],
            data[1]['prior_ty'], data[1]['prior_tz'], data[0]))
    conn.commit()

    # 关闭数据库连接
    conn.close()


def add_snapshots_prior(project_path):
    database_path = join(project_path, 'database.db')
    car_pose_txt_path = join(project_path, 'car_poses.txt')
    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()
    # 检查表是否存在
    cursor.execute("SELECT count(*) FROM sqlite_master WHERE type='table' AND name='snapshots'")
    exists = cursor.fetchone()[0] == 1
    # 打印结果
    if not exists:
        cursor.execute('''
        CREATE TABLE snapshots (
        snapshot_id INTEGER PRIMARY KEY,
        qw FLOAT,
        qx FLOAT,
        qy FLOAT,
        qz FLOAT,
        tx FLOATL,
        ty FLOAT,
        tz FLOAT
        )''')

        # 提交更改并关闭连接
        conn.commit()
    photo_info_dict = {}
    # qw,qx,qy,qz,tx,ty,tz
    # 打开文本文件
    with open(car_pose_txt_path, 'r') as file:
        image_poses = file.readlines()[::2]

    # 遍历文件行
    for image_pose in image_poses:
        data = image_pose.strip().split()
        # 解析字段
        # print(data)
        name = int(data[-1].replace('.jpg', ''))
        photo_info_dict[name] = {
            'prior_qw': float(data[1]),
            'prior_qx': float(data[2]),
            'prior_qy': float(data[3]),
            'prior_qz': float(data[4]),
            'prior_tx': float(data[5]),
            'prior_ty': float(data[6]),
            'prior_tz': float(data[7])
        }
    query = "SELECT  snapshot_id FROM snapshots"
    cursor.execute(query)
    results = cursor.fetchall()
    # print(results)
    for data in photo_info_dict.items():
        # print(data)
        # print(data[0])
        # print(data[1]['prior_qw'])
        snapshot_data = (data[0], data[1]['prior_qw'], data[1]['prior_qx'], data[1]['prior_qy'], data[1]['prior_qz'],
                         data[1]['prior_tx'], data[1]['prior_ty'], data[1]['prior_tz'])

        cursor.execute(
            '''INSERT INTO snapshots (snapshot_id, qw, qx, qy, qz, tx, ty, tz) VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
            snapshot_data)
    conn.commit()

    # 关闭数据库连接
    conn.close()


def delete_camera_data_from_db(project_path):
    # 连接到数据库
    database_path = join(project_path, 'database.db')
    conn = sqlite3.connect(database_path)

    # 创建游标对象
    cursor = conn.cursor()

    try:
        # 删除 cameras 表第7行及其以后的数据
        cursor.execute("DELETE FROM cameras WHERE prior_focal_length == 0")
        # 提交事务
        conn.commit()
        print("数据删除成功！")
    except Exception as e:
        # 发生错误时回滚操作
        conn.rollback()
        print("数据删除失败:", str(e))
    finally:
        # 关闭游标和连接
        cursor.close()
        conn.close()


def add_all_prior(project_path):
    # if if_add_images_prior:
    add_images_prior(project_path)
    add_cameras_prior(project_path)
    add_snapshots_prior(project_path)
    update_snapshot_id(project_path)
    delete_camera_data_from_db(project_path)


if __name__ == '__main__':
    add_all_prior('/dataset/rtfbag/EP40-PVS-42/EP40-PVS-42_EP40_MDC_0930_1215_2023-06-29-05-21-13_83/sfm/chouzhen')
