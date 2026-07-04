from vidgear.gears import CamGear
from vidgear.gears import WriteGear
import cv2

# Open gstreamer source
g_source = "1, width=1920, height=1080,format=NV12, framerate=30/1 ! nvvidconv ! video/x-raw,format=I420 ! appsink"
stream = CamGear(source=0, logging=True).start() 

# define required FFmpeg parameters for your writer
output_params = {"-vcodec":"libx264","-profile:v":"main","-preset:v":"veryfast","-g":60,"-keyint_min":60,"-sc_threshold":0,"-b:v":"2500k","-maxrate":"2500k","-bufsize":"2500k", "-f":"flv"}

# Define writer with defined parameters
writer = WriteGear(output = 'rstp://192.168.137.104:8089', logging =True, **output_params)

# loop over
while True:

    # read frames from stream
    frame = stream.read()

    # check for frame if Nonetype
    if frame is None:
        break


    # {do something with the frame here}


    # write frame to writer
    writer.write(frame)

    # Show output window
    cv2.imshow("Output Frame", frame)

    # check for 'q' key if pressed
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# close output window
cv2.destroyAllWindows()

# safely close video stream
stream.stop()

# safely close writer
writer.close()