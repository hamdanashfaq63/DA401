# Data

The CuZr metallic glass dataset used in the capstone (65,536 nanodiffraction patterns, 120 x 120 px, float32, `STEMcroppedRAW.tif`) was collected by the Hwang Lab at The Ohio State University and is used with permission. It is not redistributed here.

To run the pipeline without it:

* `foldsym demo` generates synthetic patterns with known fold symmetry.
* Any public 4D-STEM stack works once reshaped to `(N, H, W)` and saved as TIFF or `.npy`, then `foldsym run <file> --config v1`.

Files placed in this folder are ignored by git.
