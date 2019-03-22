package ramo.klevis.ml.yolo;

import lombok.extern.slf4j.Slf4j;
import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.Java2DFrameConverter;
import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.layers.objdetect.DetectedObject;
import org.deeplearning4j.nn.layers.objdetect.Yolo2OutputLayer;
import org.deeplearning4j.nn.layers.objdetect.YoloUtils;
import org.deeplearning4j.zoo.model.TinyYOLO;
import org.deeplearning4j.zoo.model.YOLO2;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Stack;

import static org.bytedeco.javacpp.opencv_core.*;
import static org.bytedeco.javacpp.opencv_imgproc.putText;
import static org.bytedeco.javacpp.opencv_imgproc.rectangle;

@Slf4j
public class Yolo {

    private static final double DETECTION_THRESHOLD = 0.5;
    //more accurate but slower
    public static ComputationGraph YOLO_V2_MODEL_PRE_TRAINED;
    //less accurate but faster
    public static final ComputationGraph TINY_YOLO_V2_MODEL_PRE_TRAINED;

    static {
        try {
            TINY_YOLO_V2_MODEL_PRE_TRAINED = (ComputationGraph) TinyYOLO.builder().build().initPretrained();
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    private final Stack<Frame> stack = new Stack();
    private final Speed selectedSpeed;
    private volatile List<DetectedObject> predictedObjects;
    private HashMap<Integer, String> map;
    private HashMap<String, String> groupMap;
    private ComputationGraph model;

    public Yolo(Speed selectedSpeed, boolean yolo) throws IOException {
        this.selectedSpeed = selectedSpeed;

        if (yolo) {
            //real yolo v2
            if (YOLO_V2_MODEL_PRE_TRAINED == null) {
                YOLO_V2_MODEL_PRE_TRAINED = (ComputationGraph) YOLO2.builder().build().initPretrained();
            }
            prepareYOLOLabels();
        } else {
            //tiny yolo
            prepareTinyYOLOLabels();
        }
        model = getSelectedModel(yolo);
        warmUp(selectedSpeed);
    }

    private void warmUp(Speed selectedSpeed) throws IOException {
        try {
            Yolo2OutputLayer outputLayer = (Yolo2OutputLayer) model.getOutputLayer(0);
            BufferedImage read = ImageIO.read(new File("AutonomousDriving/src/main/resources/sample.jpg"));
            INDArray indArray = prepareImage(read, selectedSpeed.width, selectedSpeed.height);
            INDArray results = model.outputSingle(indArray);
            outputLayer.getPredictedObjects(results, DETECTION_THRESHOLD);
        } catch (IOException e) {
            log.error("Failed to warm , ignoring for now",e);
        }
    }

    public void push(Frame matFrame) {
        stack.push(matFrame);
    }

    public void drawBoundingBoxesRectangles(Frame frame, Mat matFrame) {
        if (invalidData(frame, matFrame)) return;

        ArrayList<DetectedObject> detectedObjects = new ArrayList<>(predictedObjects);
        YoloUtils.nms(detectedObjects, 0.5);
        for (DetectedObject detectedObject : detectedObjects) {
            createBoundingBoxRectangle(matFrame, frame.imageWidth, frame.imageHeight, detectedObject);
        }

    }

    private boolean invalidData(Frame frame, Mat matFrame) {
        return predictedObjects == null || matFrame == null || frame == null;
    }

    public void predictBoundingBoxes() throws IOException {
        long start = System.currentTimeMillis();
        Yolo2OutputLayer outputLayer = (Yolo2OutputLayer) model.getOutputLayer(0);
        INDArray indArray = prepareImage(stack.pop(), selectedSpeed.width, selectedSpeed.height);
        log.info("stack of frames size " + stack.size());
        if (indArray == null) {
            return;
        }

        INDArray results = model.outputSingle(indArray);
        if (results == null) {
            return;
        }
        predictedObjects = outputLayer.getPredictedObjects(results, DETECTION_THRESHOLD);

        log.info("stack of predictions size " + predictedObjects.size());
        log.info("Prediction time " + (System.currentTimeMillis() - start) / 1000d);
    }

    private ComputationGraph getSelectedModel(boolean yolo) {
        ComputationGraph model;
        if (yolo) {
            model = YOLO_V2_MODEL_PRE_TRAINED;
        } else {
            model = TINY_YOLO_V2_MODEL_PRE_TRAINED;
        }
        return model;
    }

    private INDArray prepareImage(Frame frame, int width, int height) throws IOException {
        if (frame == null || frame.image == null) {
            return null;
        }
        BufferedImage convert = new Java2DFrameConverter().convert(frame);
        return prepareImage(convert, width, height);
    }

    private INDArray prepareImage(BufferedImage convert, int width, int height) throws IOException {
        NativeImageLoader loader = new NativeImageLoader(height, width, 3);
        ImagePreProcessingScaler imagePreProcessingScaler = new ImagePreProcessingScaler(0, 1);

        INDArray indArray = loader.asMatrix(convert);
        if (indArray == null) {
            return null;
        }
        imagePreProcessingScaler.transform(indArray);
        return indArray;
    }

    private void prepareYOLOLabels() {
        prepareLabels(COCO_CLASSES);
    }

    private void prepareLabels(String[] coco_classes) {
        if (map == null) {
            groupMap = new HashMap<>();
            groupMap.put("car", "Car");
            groupMap.put("bus", "Car");
            groupMap.put("truck", "Car");
            int i = 0;
            map = new HashMap<>();
            for (String s1 : coco_classes) {
                map.put(i++, s1);
                groupMap.putIfAbsent(s1, s1);
            }
        }
    }

    private void prepareTinyYOLOLabels() {
        prepareLabels(TINY_COCO_CLASSES);
    }

    private void createBoundingBoxRectangle(Mat file, int w, int h, DetectedObject obj) {

        double[] xy1 = obj.getTopLeftXY();
        double[] xy2 = obj.getBottomRightXY();
        int predictedClass = obj.getPredictedClass();
        int x1 = (int) Math.round(w * xy1[0] / selectedSpeed.gridWidth);
        int y1 = (int) Math.round(h * xy1[1] / selectedSpeed.gridHeight);
        int x2 = (int) Math.round(w * xy2[0] / selectedSpeed.gridWidth);
        int y2 = (int) Math.round(h * xy2[1] / selectedSpeed.gridHeight);
        rectangle(file, new Point(x1, y1), new Point(x2, y2), Scalar.RED);
        putText(file, groupMap.get(map.get(predictedClass)), new Point(x1 + 2, y2 - 2), FONT_HERSHEY_DUPLEX, 1, Scalar.GREEN);
    }


    private final String[] COCO_CLASSES = {"person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train",
            "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
            "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
            "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
            "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
            "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
            "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard",
            "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
            "teddy bear", "hair drier", "toothbrush"};
    private final String[] TINY_COCO_CLASSES = {"aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car",
            "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant",
            "sheep", "sofa", "train", "tvmonitor"};
}