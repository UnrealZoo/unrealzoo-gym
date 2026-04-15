import time
from unrealcv import UnrealCv_API

#before recording, open the binary and type "vget /unrealcv/status " check the port number，
#make sure the binary port is consistent with the number in the "port=****"
api = UnrealCv_API(port=9000, ip='127.0.0.1', resolution=(640, 480))

cam_id = 0

api.set_warmup_frames(10)

# Start recording a 5-second clip at 30 FPS
# Record options can be: 'lit', 'mask', 'depth', 'normal', 'flow', etc.
output_folder = 'C:/recordings/scene01'
fps = 30
duration = 5.0


print('Recording completed!')