# A Hyperspectral Underwater Target Detection Method Using Bathymetric Model
-----------

### Acknowledgements

This code is modified from [TUTDF](https://github.com/lizheyong/TUTDF). I have adjusted and modified the code to make it runnable with a single command.
> *Reference:Li, Zheyong, et al. "A transfer-based framework for underwater target detection from hyperspectral imagery." Remote Sensing. 2023.*

Hyperspectral underwater target detection is a promising and challenging task in remote sensing image processing. The main difference between hyperspectral underwater target detection and hyperspectral land-based target detection is the spectral distortion caused by the underwater environment. Addressing this distortion is the primary challenge in hyperspectral underwater target detection.


<!DOCTYPE html>
<html>
<head>
    <!-- <script type="text/javascript" id="MathJax-script" async
      src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js">
    </script> -->
</head>
<body>

<p>We utilize the following bathymetric model to synthetic underwater target spectra using for training.</p>

$$
r(\lambda) = r_{\infty}(\lambda) \left( 1 - e^{-(k_d(\lambda) + k^c_u(\lambda))H} \right) + \frac{r_B(\lambda)}{\pi} e^{-(k_d(\lambda) + k^b_u(\lambda))H}
$$

</body>
</html>

> *Reference:Lee, Zhongping, et al. "Hyperspectral remote sensing for shallow waters. I. A semianalytical model." Applied optics. 1998.*

The detection algorithm framework is shown in the figure below.
<p align="center">
  <img src="/pics/framework.png" alt="Framework" title="Transfer-UTD" width="900px">
</p>

To solve the above bathymetric model, we need to estimate the inherent optical properties (IOPs) of the water. Here, we use IOPE-Net for this purpose. Which is implemented by [lizheyong](https://github.com/lizheyong/IOPE-Net). 
> *Reference:Qi, Jiahao, et al. "Hybrid sequence networks for unsupervised water properties estimation from hyperspectral imagery." IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing. 2021.*

Then, use the bathymetric model to synthesize underwater target spectra as positive samples and randomly select spectra as negative samples for training. Finally, test on the input hyperspectral images.

***
### Dataset
Due to the difficulty of deploying underwater targets and the high cost of data collection, research in this area has predominantly relied on simulated data. To advance the study of underwater target detection in real-world scenarios, we collected a dataset of real underwater scenes and conducted experiments on this data. The deployed underwater target is an iron plate, and the target's prior spectral data were collected onshore.

<p align="center">
  <img src="/pics/ref.png" alt="Framework" title="Transfer-UTD" width="400px">
</p>

> The River Scene data sets was captured by Headwall Nano-Hyperspec imaging sensor equipped on DJI Matrice 300 RTK unmanned aerial vehicle, and it was collected at the Qianlu Lake Reservoir in Liuyang (28◦18′40.29′′ N, 113◦21′16.23′′ E), Hunan Province, China on July 31, 2021.


- **Download the datasets from [*here*](https://drive.google.com/file/d/1UmZkrI-5pA0h6zF5HRrksaXmwLap0ipA/view?usp=sharing), put it under the folder <u>dataset</u>.**
  
- Dataset format: mat

<p align="center">
  <img src="/pics/datasets.png" alt="Framework" title="Transfer-UTD" width="800px">
</p>

- River Scene1:
242×341 pixels with 270 spectral bands

- River Scene2:
255 × 261 pixels with 270 spectral bands

- River Scene3:
137 × 178 pixels with 270 spectral bands

- Simulated Data:
The data set has a spatial resolution of 200 × 200 pixels, with wavelength coverage from 400 to 700 nm at 150 spectral bands.

- Ningxiang & Ningxiang2:
The Ningxiang dataset was collected in a reservoir with high mineral content and significant sediment, which may make detection challenging.

Keys: 
- data: The hyperspectral imagery contains underwater targets
- target: The target prior spectrum collected on land
- gt: The ground truth of underwater target distribution

----

### Training

1. Modify `train.py`
2. Run ` python train.py `

Training for new dataset need to generate NDWI mask (If land areas are included). 
> NDWI Water Mask (require gdal):
> `water_mask\NDWI.py`
> - water -- 0
> - land -- 255
> - selected bands get from envi
> - GREEN.tif: green band 549.1280 nm
> - NIR.tif: near-infrared band 941.3450 nm

### Testing

1. Modify `pred.py`
2. Run ` python pred.py `

> You can download my model weights from [*here*](https://drive.google.com/file/d/18-dQ5kppNuky8cwQ8TwtftVuab1B6D4h/view?usp=sharing), put it in data_temp folder.

### Note

According to the implementation of the original code, the number of neurons in the linear1 layer specified in the `preprocess.py` file may need to be adjusted.

---

*There has been limited research in this field, and many challenges remain in applying these methods to real-world scenarios. We sincerely hope that this work contributes positively to the field, despite the theoretical and practical limitations that still exist. If you have any concerns, please do not hesitate to contact liu_jiaxuan2021@163.com.*

