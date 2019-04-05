

# Installation

First, you need to compile the Retro native library. This 
can be done by executing the following commands:

```bash
cd <WORKING_DIRECTORY>
git clone git@github.com:eaplatanios/retro.git
cd retro
git checkout c-api
cmake . -G 'Unix Makefiles' -DBUILD_PYTHON=OFF -DBUILD_C=ON
make -j4 retro-c
```

This will result in a `libretro.dylib` file in the working 
directory and in compiled core files for multiple gaming 
platforms in `<WORKING_DIRECTORY>/retro/cores`, among other 
things.
