## Dashboard

### model

* hard attention: auto select threshold in common.utils.BoundingBoxCrop
* soft attention. The hard one is out of expediency. But how to apply an weight on the original image w/o modify inference path?
* debug of focal and smooth loss. Maybe no bug at all.
* debug of dice coefficient
* online fuse(augment) using segment and input, with random nature.
* a precise discriminator. Not a toy.
* blur pooling(downsample) in unet.downconv

### engineering

* make a tester/predicter
* 📌 batch&data-wise meta
* a base to manage losses?
* diverse augments, both random arg aug and more kinds of augment.
* multi-process in dataloader
* profile the dataloader
