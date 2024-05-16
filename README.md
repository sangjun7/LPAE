# LPAE
This repository is for Laplacin Pyramid-like Auto-Encoder (LPAE) introduced in the following paper.

Sangjun Han, Taeil Hur, Youngmi Hur, "Laplacian Pyramid-like Autoencoder", In: Arai, K. (eds) Intelligent Computing. SAI 2022. Lecture Notes in Networks and Systems, vol 507. Springer, Cham. https://doi.org/10.1007/978-3-031-10464-0_5  
(arXiv version : https://doi.org/10.48550/arXiv.2208.12484)

The pretrained parameters are provided by Google Drive. 
https://drive.google.com/drive/folders/1uVY4yn3K4n-2EWi1A9f0mUbQdaHgntXQ?usp=sharing

## Test
We provide just one test code `div2k_srnet/test_real_img.py` for the super-resolution experiment of LPSR trained on DIV2K.
This code takes images in a target directory as input, enlarges them by 2, 4, or 8 times, and then saves them.
Generalization of this code or code for other models will be posted later.

To run the code, use the following command:
```
python scripts/chatlean_bfs.py --API_key [OpenAI API key] --minif2f [Path for minif2f dataset] --model [Model name] --temperature [Temperature] --ex_data datasets/prompt_examples/examples.json --num_sample [Number of Samples] --result_fname [Name of result file]
```
