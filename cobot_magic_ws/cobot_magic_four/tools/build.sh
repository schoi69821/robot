ws_path=$(pwd)
echo $ws_path

echo "build camera"
cd $ws_path/camera_ws
catkin_make
cd ..

echo "build follow_control"
cd $ws_path/follow_control
./tools/build.sh
cd ..

echo "print camera serial"
tools/camera_serial.sh