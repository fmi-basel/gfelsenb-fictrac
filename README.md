<div align="left">
  <img src="http://rjdmoore.net/fictrac/header_text.jpg"><br><br>
</div>

**FicTrac** is an open source software library and set of executables for reconstructing the fictive path of an animal walking on a patterned sphere. The software is fast, flexible, and easy to use and simplifies the setup of closed-loop tracking experiments.

FicTrac was originally developed by researchers at the [Queensland Brain Institute](http://qbi.uq.edu.au/) at the University of Queensland, Australia for tracking honeybees and fruit flies during closed-loop tethered walking experiments, but it has since proved useful for tracking a wide range of animals with different movement speeds, track ball diameters and patterns, and stimuli.

On this page you'll find information for:
* [Getting started](#getting-started) - including [hardware requirements](#experimental-setup), and basic steps for [installing](#installation), [configuring](#configuration), and [running FicTrac](#running-fictrac).
* [Research](#research) - including works to cite and links to the original FicTrac publication and preprint (pdf).
* [Contributing](#contribution-guidelines) - how to go about making your fixes, additions, and customisations available for other users.
* [Re-use](#license) - licensing information, if you plan on using FicTrac or the source code.

You might also be interested in the following links:
* [Demo video](http://youtu.be/BeGYOEOdWjw) - Quick (30s) overview of what FicTrac does and how it works.
* [FicTrac manual](http://link) - Detailed instructions, description of output data, parameters, recommendations, etc.
* [Homepage](http://fictrac.rjdmoore.net) - Contact details for the main author/developer, links, and further info.
* [Forum](http://www.reddit.com/r/fictrac/) - Subreddit for FicTrac users to share issues and advice.
* [Mailing list](http://fictrac.rjdmoore.net/mail.html) - Subscribe to receive important announcements and updates.

Happy tracking!

## Getting started

If you're just setting up your lab, or wondering whether FicTrac is suitable for your setup (spoiler: yes, probably), check the [section below](#experimental-setup) for the basic requirements.

If you already have an experimental enclosure with a camera, you can use FicTrac to either process those videos offline or to run live from the camera. Just follow the sections below to [install](#installation), [configure](#configuration), and [run FicTrac](#running-fictrac).

### Experimental setup

Very briefly, FicTrac imposes almost no special requirements on your experimental setup, other than that a pattern must be applied to the track ball. However, there are a number of tips that can help you get the best results from using FicTrac.

A typical FicTrac experimental setup might include at least the following equipment:
* *Animal tether/harness* - for keeping the animal stationary on the track ball
* *Stimulus* - sensory feedback for the animal
* *Track ball support* - structure to hold track ball in place whilst allowing free rotation
* [Track ball](doc/requirements.md#track-ball) - lightweight sphere that is rotated by the animal's walking/turning motions. Surface pattern should ideally be high-contrast, non-repeating, non-reflective (not glossy), and contain around 10~50 variously sized blobby shapes.
* [Video camera](doc/requirements.md#video-camera) - for monitoring the track ball (and animal). Resolution is not important, and for vertebrates a USB webcam is likely sufficient. For faster moving insects, the camera should support frame rates >100 Hz. In all cases, the frame rate, exposure, and lens should be chosen to ensure the track ball pattern is well-exposed under all lighting/stimulus conditions and that there is no motion blur during the fastest expected movements. At least one half of one hemisphere of the track ball surface should be visible by the camera.
* [PC/laptop](doc/requirements.md#pclaptop) - for running FicTrac software (and generating closed-loop stimuli). Processor should be somewhat recent (>2 GHz, multi-core).
* [Lighting](doc/requirements.md#lighting) - ambient lighting should be diffuse (no specular reflections from track ball surface) and bright enough to give good track ball surface exposure at chosen frame rate.

FicTrac imposes no requirements on the *italicised* items; how you design these is completely dependent on other factors.

### Installation

The FicTrac source code can be built for both Windows and Ubuntu (Linux) operating systems. You can even build and run FicTrac from within a virtual machine on any operating system.

#### Windows

1. Download and install required dependencies:
    1. [Cmake build system](https://cmake.org/download/) (binary distribution)
    2. [OpenCV computer vision library](https://opencv.org/releases.html) (latest release Win pack)
    3. [NLopt optimisation library](https://nlopt.readthedocs.io/en/latest/NLopt_on_Windows/) (precompiled DLL)
2. Clone or download the FicTrac repository, then navigate to that folder, open a terminal, and create a build directory:
```
mkdir build
cd build
```
3. Next, we will configure and build the FicTrac project. FicTrac is written in C++, so you'll need a suitable compiler. In this example we will use MSVS Build Tools. If you don't already have Visual Studio, you will need to install the [build tools](https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2017).
4. Run Cmake to prepare the necessary build files for FicTrac. Here we also need to provide the paths to where we installed OpenCV and NLopt (I have given example paths here, you will need to modify them for your installation):
```
cmake -G "Visual Studio 15 2017 Win64" -D OPENCV_DIR="C:\path\to\opencv-3.4.2\build" -D NLOPT_DIR="C:\path\to\nlopt-2.4.2\" ..
```
5. Finally, build and install FicTrac:
```
cmake --build . --config Release --target ALL_BUILD
```

If everything went well, the executables for FicTrac and a configuration utility will be placed in the `bin` directory under the FicTrac source folder.

**Note:** To save video, you must also have the [H264 library](https://github.com/cisco/openh264/releases). OpenCV 3.4.2 requires openh264-1.7.0-win64.dll; if you run FicTrac with the wrong H264 library version, OpenCV should print an error message to terminal. Simply copy the .dll to your system path, or to the same folder as `fictrac.exe`.

#### Ubuntu (Linux)

1. Install the required dependencies:
```
sudo apt-get install gcc cmake libavcodec-dev libavformat-dev libswscale-dev libv4l-dev libgtk-3-dev libdc1394-22-dev libopencv-dev libnlopt-dev
```
2. Clone or download the FicTrac repository, then navigate to that folder and create a build directory:
```
mkdir build
cd build
```
3. Run Cmake to prepare the necessary build files for FicTrac (if OpenCV and NLopt are not installed in the default location, you can help Cmake find them by defining OPENCV_DIR and NLOPT_DIR - see [Windows installation](#windows) for an example):
```
cmake ..
```
4. Finally, build and install FicTrac:
```
make -j4
```

If everything went well, the executables for FicTrac and a configuration utility will be placed in the `bin` directory under the FicTrac source folder.

### Configuration

There are two neccessary steps to configure FicTrac prior to running the program:
1. You must provide a configuration text file that contains important parameters for your setup. At a minimum, this config file must define the parameters `src_fn` and `vfov`, which are the path to the image source (video file or camera) and vertical field of view (in degrees) of your camera/lens respectively. If you are running live from the camera, then `src_fn`is the camera index (e.g. 0). **The vertical field of view for your camera/lens must be specified accurately.** If `vfov` is incorrect, the surface map created by FicTrac will not wrap around the sphere correctly, and tracking will fail. An example config file is provided in the `sample` directory under the FicTrac source folder; you can use this file as a template to write your own config file.
2. You must run the interactive configuration program (configGui), passing the path to your config file (above) as an argument. This program will guide you through the configuration of the track ball region of interest within your input images and the transformation between the camera's and animal's frames of reference. **It is important that the camera is not moved after running the configuration utility. If the camera is moved, you must reconfigure.** After running the configuration utility, your config file will have some additional default parameters added automatically.

A more detailed guide for configuring FicTrac for your setup, as well as a complete list and explanation of parameters that can be specified, can be found in the [FicTrac manual]().

The commands for running the configuration utility under Windows and Ubuntu (Linux) are almost identical. Simply open a terminal in the FicTrac source folder and type:

#### Windows
```
.\bin\Release\configGui.exe path\to\config.txt
```

#### Ubuntu (Linux)
```
./bin/configGui path/to/config.txt
```

### Running FicTrac

Once you have configured FicTrac for your setup, you may run FicTrac simply by opening a terminal in the FicTrac source folder and typing:

#### Windows
```
.\bin\Release\fictrac.exe path\to\config.txt
```

#### Ubuntu (Linux)
We execute as super user so that we can set the FicTrac process to higher priority than other system processes.
```
sudo ./bin/fictrac path/to/config.txt
```

## Research

If you use FicTrac as part of your research, please cite the original FicTrac publication:

> RJD Moore, GJ Taylor, AC Paulk, T Pearson, B van Swinderen, MV Srinivasan (2014). *"FicTrac: a visual method for tracking spherical motion and generating fictive animal paths"*, Journal of Neuroscience Methods, Volume 225, 30th March 2014, Pages 106-119. [[J. Neuroscience Methods link]]() [[Preprint (pdf) link]]().

This publication contains technical details on how FicTrac works, performance analysis, results, and other discussion.

## Contribution guidelines

If you have modified the FicTrac source code to fix issues, add functionality, or to better suit your setup - please consider making those additions available to other users!

To do so, just follow the standard [Github fork and pull request workflow](https://gist.github.com/rjdmoore/ed014fba0ee2c7e75060ccd01b726cb8).

## License

[CC BY-NC-SA 3.0](LICENSE.txt)
