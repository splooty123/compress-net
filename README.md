This is a neural network thing I made so I can have good cool science fair.
It runs image auto-encoder because I want to do image compression.

Here some of output:
<img width="3200" height="144" alt="grid_output" src="https://github.com/user-attachments/assets/5968e21a-2ff4-4d22-b592-1f1367a9288b" />

Pretty cool.

Run it by running it. It will ask for image output. If say 1 it will output occasional .bmp image to image_output folder.
It trains off image in images folder. At the end it will output losses so can graph or something.
Input images can be any format supported by the bundled `stb_image.h`, like JPG, PNG, BMP, TGA, GIF, HDR, PSD, PIC, and PNM.

## Build

Cross-platform CMake build:

```sh
cmake -S . -B build
cmake --build build
```

Then run the executable from the project root so it can find `images/`:

```sh
./build/compress_net
```

On Windows, run `build\Debug\compress_net.exe` or `build\Release\compress_net.exe` depending on your generator/configuration.

OpenMP is optional. CMake will use it when it is available, and the program still builds without it.
