#! /bin/bash

/usr/bin/xrandr -d :0 --output VIRTUAL1 --primary --auto
/usr/bin/xrandr --newmode "1600x900_60.00" 118.25 1600 1696 1856 2112 900$
/usr/bin/xrandr --addmode VIRTUAL1 "1600x900_60.00"
/usr/bin/xrandr
