# DCGAN Anime Generation
Tried out vanilla GAN and LSGAN for anime generation using DCGAN structure.

## Training process
- Used original DCGAN paper settings.
- Halved the discriminator loss to balance G/D gradient descent schedule.
- Used soft-labeling (0.8/0.9 ~ 1)

## Results
### Vanilla GAN
### LSGAN