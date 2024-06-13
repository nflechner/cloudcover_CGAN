### Architecture

This version of the CGAN has 1 generator and two discriminators: the spatial and temporal discriminators. The spatial discriminator takes as input only 1 image, meaning that it gives losses that correspond to 'how real does this single image look'? The temporal discriminator on the other hand takes as input satellite image sequences, such that losses correspond to 'how real does this sequence look?'. 

Additionally, there is a conditioning stack, which entails that context images are fed to the generator at multiple resolutions throughout the upsampling process. 

### Losses

In this variant, the generator and discriminator both learn from hinge loss. Additionally, the generator receives a grid cell regularizing term, which is added to the loss before these are backpropagated. 

### Choices

In this version with the conditioning stack, it appeared that the generator was quite strong and the discriminator needed to become stronger. Furthermore, in later epochs the images became blurry, so we wanted to implement some losses that would improve learning. Therefore, the binary crossentropy loss that was previously used was exchanged for the hinge loss, which has the possibility of having a higher accuracy than a logistic based loss by being less prone to outliers. 