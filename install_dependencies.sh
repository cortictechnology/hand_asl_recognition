sudo apt-get update && sudo apt-get upgrade -y
sudo apt-get install -y libusb-1.0-0-dev \
                        libjpeg-dev \
                        libtiff5-dev \
                        libjasper-dev \
                        libpng-dev \
                        libavcodec-dev \
                        libavformat-dev \
                        libswscale-dev \
                        libv4l-dev \
                        libxvidcore-dev \
                        libx264-dev \
                        libfontconfig1-dev \
                        libcairo2-dev \
                        libgdk-pixbuf2.0-dev \
                        libpango1.0-dev \
                        libgtk2.0-dev \
                        libgtk-3-dev \
                        libatlas-base-dev \
                        gfortran \
                        libhdf5-dev \
                        libhdf5-serial-dev \
                        libhdf5-103 \
                        libqtgui4 \
                        libqtwebkit4 \
                        libqt4-test \
                        python3-pyqt5

sudo pip3 install -r requirements.txt
echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="03e7", MODE="0666"' | sudo tee /etc/udev/rules.d/80-movidius.rules
sudo udevadm control --reload-rules && sudo udevadm trigger

echo "Dependencies are all installed, it is recommended to unplug and plug in your OAK-D device after this."