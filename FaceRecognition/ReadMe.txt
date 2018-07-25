The implementation is based on https://github.com/iwantooxxoox/Keras-OpenFace

In this application the weights loading so the transfer learning is a bit different.
We are using the wights from keras saved into .csv files and than fit that into DL4J structure
In future further consolidation may need in the way we load the wights so right not the model may
still need some tuning so please time to time check in the code
as it will continually improved to a state of the art accuracy


Please notice that the open face model is quite small comparing to real systems so the accuracy may not
be the best but it is quite promising and it clearly shows that the concept explored in slides are working