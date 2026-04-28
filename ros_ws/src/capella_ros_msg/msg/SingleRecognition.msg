uint8 FIRE=0 # 火焰
uint8 SHIT=1 # 粪便
uint8 class_id # 类别
string uuid #检测目标唯一编码，防止误识别
sensor_msgs/Image image
int16 left_cord_x #检测框左上角坐标x
int16 left_cord_y #检测框左上角坐标y
int16 right_cord_x #检测框右下角坐标x
int16 right_cord_y #检测框右下角坐标y