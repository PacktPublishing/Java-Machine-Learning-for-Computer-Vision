package ramo.klevis.ml.yolo;

import lombok.extern.slf4j.Slf4j;
import org.bytedeco.javacpp.opencv_core;
import org.bytedeco.javacv.FFmpegFrameGrabber;
import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.FrameGrabber;
import org.bytedeco.javacv.OpenCVFrameConverter;

import java.util.concurrent.atomic.AtomicInteger;

import static org.bytedeco.javacpp.opencv_highgui.*;

@Slf4j
public class VideoPlayer {

    private static final String AUTONOMOUS_DRIVING_RAMOK_TECH = "Autonomous Driving(ramok.tech)";
    private String windowName;
    private volatile boolean stop = false;
    private Yolo yolo;
    private final OpenCVFrameConverter.ToMat converter = new OpenCVFrameConverter.ToMat();
    public final static AtomicInteger atomicInteger = new AtomicInteger();


    public void startRealTimeVideoDetection(String videoFileName, Speed selectedIndex, boolean yoloModel) throws java.lang.Exception {
        log.info("Start detecting video " + videoFileName);
        int id = atomicInteger.incrementAndGet();
        windowName = AUTONOMOUS_DRIVING_RAMOK_TECH + id;
        log.info(windowName);
        yolo = new Yolo(selectedIndex, yoloModel);
        startYoloThread();
        runVideoMainThread(videoFileName, converter);
    }

    private void runVideoMainThread(String videoFileName, OpenCVFrameConverter.ToMat toMat) throws FrameGrabber.Exception {
        FFmpegFrameGrabber grabber = initFrameGrabber(videoFileName);
        while (!stop) {
            Frame frame = grabber.grab();
            if (frame == null) {
                log.info("Stopping");
                stop();
                break;
            }
            if (frame.image == null) {
                continue;
            }
            yolo.push(frame);
            opencv_core.Mat mat = toMat.convert(frame);
            yolo.drawBoundingBoxesRectangles(frame, mat);
            imshow(windowName, mat);
            char key = (char) waitKey(20);
            // Exit this loop on escape:
            if (key == 27) {
                stop();
                break;
            }
        }
    }

    private FFmpegFrameGrabber initFrameGrabber(String videoFileName) throws FrameGrabber.Exception {
        FFmpegFrameGrabber grabber = new FFmpegFrameGrabber(videoFileName);
        grabber.start();
        return grabber;
    }

    private void startYoloThread() {
        Thread thread = new Thread(() -> {
            while (!stop) {
                try {
                    yolo.predictBoundingBoxes();
                } catch (Exception e) {
                    //ignoring a thread failure
                    //it may fail because the frame may be long gone when thread get chance to execute
                }
            }
            yolo = null;
            log.info("YOLO Thread Exit");
        });
        thread.start();
    }

    public void stop() {
        if (!stop) {
            stop = true;
            destroyAllWindows();
        }
    }
}