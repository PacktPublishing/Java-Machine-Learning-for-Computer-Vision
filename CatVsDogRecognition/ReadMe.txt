Please note that the Model was trained with only cats and dogs.
So since it has not seen any other than cats or dogs the model tends to perform not good with
images different than cats or dogs

Anyway the TransferLearningVGG16.train can easily adapted to train
with additional class(general images) called maybe 'non cat or dog'

Even further you may decide to replace the training photos(DATA_PATH) completely and the number of classes so
then the network we will easily be extended to detect other images
On CPU depending on data it may take several hours while on GPU half to one hour at maximum
Please increase the memory since VGG16 is really a big network