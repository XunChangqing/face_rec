# face_rec
face register and recognize with opencv

build & run:
mkdir build
cd build
cmake ..
./face\_reg --dir=/path/to/save/faces person\_name1
Press space to store one frame, Press Esc to quit.
./face\_reg --dir=/path/to/save/faces person\_name2
...

./face\_rec /path/to/save/faces
