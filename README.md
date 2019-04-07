**WARNING:** This is currently work-in-progress and is 
changing rapidly. I expect it to be more stable within 
about a week or two.

# Prerequisites

In order to use the image renderer you need to first 
install GLFW. You can do so, as follows:

```bash
# For MacOS:
brew install --HEAD git glfw3

# For Linux:
sudo apt install libglfw3-dev libglfw3
```

Then, in order to use it you need to provide the following
extra flags when using `swift build` or any other SwiftPM 
command: `-Xcc -DGLFW -Xswiftc -DGLFW`. Furthermore, if 
`libglfw.so` or `libglfw.dylib` is not in your 
`LD_LIBRARY_PATH`, you also need to provide the following 
flags: `-Xlinker -lglfw -Xlinker -L<path>`, where `<path>` 
represents the path to the dynamic library. For example:

```bash
swift test \
  -Xcc -DGLFW -Xswiftc -DGLFW \
  -Xlinker -lglfw -Xlinker -L/usr/local/lib
```

**Note:** If the rendered image does not update according 
to the specified frames per second value and you are using 
MacOS 10.14, you should update to 10.14.4 because there is 
a bug in previous releases of 10.14 which breaks VSync.

# Installation

If `libretro.so` or `libretro.dylib` is not in your 
`LD_LIBRARY_PATH`, you need to provide the following 
extra flags when using `swift build` or any other SwiftPM 
command: `-Xlinker -L<path>`, where `<path>` represents the 
path to the dynamic library. The simplest way to start is 
to execute the following commands from within your code 
directory:

```bash
git clone git@github.com:eaplatanios/retro.git
cd retro
git checkout c-api
cmake . -G 'Unix Makefiles' -DBUILD_PYTHON=OFF -DBUILD_C=ON
make -j4 retro-c
```

This will result in a `libretro.so` or `libretro.dylib` 
file in the `retro` subdirectory and in compiled core files 
for multiple gaming platforms in the `retro/cores`
subdirectory.
