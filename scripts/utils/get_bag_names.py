import pandas as pd

# 读取Excel文件
df = pd.read_excel('/dataset/maptr_data/appen_data/20230831/pgo_pose_global_imu_pose_evo_all.xlsx', header=None)

# 提取第一列的字符串内容（从第二行开始）
strings = df.iloc[1:, 0].tolist()

# 格式化字符串列表为 ['xxx', 'xxx'] 的格式
formatted_strings = ["'{}'".format(s) for s in strings]

# 将格式化后的字符串列表保存到文本文件
with open('output.txt', 'w') as file:
    file.write('[' + ',\n '.join(formatted_strings) + ']')

# 打印成功消息
print('字符串已保存到 output.txt 文件中。')