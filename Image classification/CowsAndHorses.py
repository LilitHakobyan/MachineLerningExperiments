
from fastai.vision import *
from fastai.metrics import error_rate

bs = 32

path = untar_data('https://n055nom5.gradient.paperspace.com/tree/course-v3/nbs/dl1/data/MyDataSet');
path

tfms = get_transforms(do_flip=False)
data = ImageDataBunch.from_folder(path, ds_tfms=tfms, size=24,bs=bs//2).normalize(imagenet_stats)

learn = create_cnn(data, models.resnet50, metrics=error_rate)

learn.lr_find()
learn.recorder.plot()

learn.fit_one_cycle(8)

learn.save('stage-0-50');

interp = ClassificationInterpretation.from_learner(learn)

interp.most_confused(min_val=-1)

interp.plot_confusion_matrix(figsize=(12,12),dpi=60)

data.show_batch(rows=4, figsize=(10,10))

interp.plot_top_losses(3,largest=True, figsize=(12, 12))

