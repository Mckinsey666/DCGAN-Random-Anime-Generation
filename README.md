# DCGAN Anime Generation
- Tried out vanilla GAN and LSGAN for anime generation using DCGAN structure.
- Collected our own dataset (scraped images from a online manga store)

|Given dataset|Our own dataset|
|-------------|---------------|
|![](./img_src/given.png)|![](./img_src/self.png)|


## Training process
- Used original DCGAN paper settings.
- Halved the discriminator loss to balance G/D gradient descent schedule.
- Used soft-labeling (0.8/0.9 ~ 1)

## Results
### Vanilla GAN
### LSGAN