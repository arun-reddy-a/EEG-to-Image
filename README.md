This repository contains code for adapting stable diffusion for EEG to Image task.

The rightmost column is the ground truth image for an EEG Signal and the remaining columns are the generated images for that particular EEG signal.

This model was trained with limited number of iterations [ran out of GPU credits]. Further Improvements are definately possible by simply training for more epochs.

![](image_570.png)

The training also exhibits sudden convergence phenomenon. The fit of the model was pretty bad until 240 iterations but suddenly within the next 10 iterations the model drastically improves as shown in the images below.

<div align="center">
  <div style="display: inline-block; text-align: center;">
    <img src="image_240.png" alt="Image 1" width="250" />
    <p>Caption for Image 1</p>
  </div>
  <div style="display: inline-block; text-align: center;">
    <img src="image_250.png" alt="Image 2" width="250" />
    <p>Caption for Image 2</p>
  </div>
  <div style="display: inline-block; text-align: center;">
    <img src="image_260.png" alt="Image 3" width="250" />
    <p>Caption for Image 3</p>
  </div>
</div>
