# Cross-domain-Anomaly-Injection
Implement of "Stones from Other Hills can Polish Jade'':  Zero-shot Anomaly Synthesis via Cross-domain Anomaly Injection

## CAI: Cross-domain Anomaly Injection

image_path, mask_path and src_path, which indicates the patch of target normal images, foreground mask of normal
image and our domain-agnostic anomaly dataset respectively, then cross-domain anomalies are saved in save_path.

multiprocess_of_CAI.py is the multiprocess version of CAI, CAI_utils_functions.py is the utility functions.

## CDM: CAI-guided Diffusion Mechanism

main.py and tarin_mask.py are the codes for training the anomaly generation model and the mask generation model.
mask_image_generator.py can generate anomaly mask-image pairs based on the mask generation model and anomaly generation
model automatically, which calls both generate_mask.py and generate_with_mask.py.

## source of real anomalies

Due to the copyright issue, we do not provide the entire dataset. So, we provide the sources of online search and reality photographing.

## Acknowledgements

- Our CAI is built based on the possion image edit implemented
  by [OpenCV](https://github.com/opencv/opencv/blob/4.x/samples/cpp/tutorial_code/photo/seamless_cloning/cloning_demo.cpp).
- Our CDM is built based on the FSAS project [Anomalydiffusion](https://github.com/sjtuplayer/anomalydiffusion).
