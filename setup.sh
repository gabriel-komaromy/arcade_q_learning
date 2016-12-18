# This is what I used to set up the environment on a new Ubuntu machine.
# This may not be complete. I recommend running it line-by-line in a terminal.
sudo apt-get install python-pip
sudo pip install --upgrade pip
sudo pip install tensorflow
sudo pip install scipy
sudo pip install Pillow
git clone https://github.com/mgbellemare/Arcade-Learning-Environment.git
cd Arcade-Learning-Environment/
sudo apt-get install libsdl1.2-dev libsdl-gfx1.2-dev libsdl-image1.2-dev cmake -y
mkdir build
cd build
cmake -DUSE_SDL=ON -DUSE_RLGLUE=OFF -DBUILD_EXAMPLES=ON ..
make -j 4
cd ..
sudo pip install .
cd ..
