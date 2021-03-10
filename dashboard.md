## Dashboard

### before merging back

* ~~📌 fix lr warm-up support~~

### model

* hard attention: auto select threshold in common.utils.BoundingBoxCrop
* soft attention. The hard one is out of expediency. But how to apply an weight on the original image w/o modify inference path?
* use triplet or other substitution. Maybe a triplet-supported sampler is needed.
* debug of focal and smooth loss. Maybe no bug at all.
* a precise discriminator. Not a toy.
* blur pooling(downsample) in unet.downconv

### engineering

* make a tester/predicter
* 📌 data-wise meta
* diverse augments, both random arg aug and more kinds of augment.
* multi-process in dataloader
* profile the dataloader
