package ramo.klevis.ml.tracking.yolo;

import org.bytedeco.javacpp.opencv_core;
import org.deeplearning4j.nn.layers.objdetect.DetectedObject;
import org.jetbrains.annotations.NotNull;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.concurrent.atomic.AtomicInteger;

public class MarkedObject implements Comparable<MarkedObject> {

    private static final AtomicInteger ID = new AtomicInteger();
    private final long created;
    private final opencv_core.Mat matFrame;
    private DetectedObject detectedObject;
    private volatile INDArray l2Norm;
    private String id;
    private volatile boolean showed;

    public MarkedObject(DetectedObject detectedObject, INDArray l2Norm, long created, opencv_core.Mat matFrame) {
        this.detectedObject = detectedObject;
        this.l2Norm = l2Norm;
        this.created = created;
        this.matFrame = matFrame;
        id = "" + ID.incrementAndGet();
    }

    public opencv_core.Mat getMatFrame() {
        return matFrame;
    }

    public DetectedObject getDetectedObject() {
        return detectedObject;
    }

    public long getCreated() {
        return created;
    }

    public INDArray getL2Norm() {
        return l2Norm;
    }

    public String getId() {
        return id;
    }

    public void setId(String id) {
        this.id = id;
    }

    @Override
    public String toString() {
        return "MarkedObject{" +
                "created=" + created +
                ", detectedObject=" + detectedObject +
                ", l2Norm=" + l2Norm +
                ", id='" + id + '\'' +
                '}';
    }

    public boolean isShowed() {
        return showed;
    }

    public void setShowed(boolean showed) {
        this.showed = showed;
    }

    @Override
    public int compareTo(@NotNull MarkedObject o) {
        return Long.compare(o.created, created);
    }
}
