# 垃圾相对机器人的位姿
geometry_msgs/PoseStamped pose

# 湿垃圾
uint8 WET_GARBAGE=1

# 干垃圾
uint8 DRY_GARBAGE=2

# 垃圾类别
int8 class_id

# 垃圾的边界框角点
geometry_msgs/Point[] bbox_corner_points