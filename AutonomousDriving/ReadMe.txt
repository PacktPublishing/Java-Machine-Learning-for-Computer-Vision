Although application architecture was tune and optimized to work on CPU, to have true real time
performance you may need to run on GPU

The application is based on two main models:

TinyYOLO a mini version or real YOLOV2 which is not very accurate but it is really fast

YOLO this is a quite good version and the accuracy is really high and impesive but it is slow
since we need to execute more weights

Beside choosing one of the models we can also choose one of the three modes:
   FAST("Real-Time but low accuracy", 224, 224, 7, 7),
   MEDIUM("Almost Real-time and medium accuracy", 416, 416, 13, 13),
   SLOW("Slowest but high accuracy", 608, 608, 19, 19);

basically as we move from FAST to SLOW the resolution increases together with the grid size.
Than means that for sure the accuracy will greatly improved but it will be quite more slow and fare
from real time