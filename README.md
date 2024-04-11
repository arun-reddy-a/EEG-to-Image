This repository contains code for adapting stable diffusion for EEG to Image task.

First Clone the official latent-diffusion repository(https://github.com/CompVis/latent-diffusion) then put this code in the scripts folder.

The rightmost column is the ground truth image for an EEG Signal and the remaining columns are the generated images for that particular EEG signal.

This model was trained with limited number of iterations [ran out of GPU credits]. Further Improvements are definitely possible by simply training for more epochs.

![](image_570.png)

The training also exhibits sudden convergence phenomenon. The fit of the model was pretty bad until 240 iterations but suddenly within the next 10 iterations the model drastically improves as shown in the images below.

<div align="center">
  <div style="display: inline-block; text-align: center;">
    <img src="image_240.png" alt="fit at iteration 240" width="250" />
    <p>fit at iteration 240</p>
  </div>
  <div style="display: inline-block; text-align: center;">
    <img src="image_250.png" alt="fit at iteration 250" width="250" />
    <p>fit at iteration 250</p>
  </div>
  <div style="display: inline-block; text-align: center;">
    <img src="image_260.png" alt="fit at iteration 260" width="250" />
    <p>fit at iteration 260</p>
  </div>
</div>
