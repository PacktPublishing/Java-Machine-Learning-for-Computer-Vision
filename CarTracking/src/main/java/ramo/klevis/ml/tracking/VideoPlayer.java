package ramo.klevis.ml.tracking;

import lombok.extern.slf4j.Slf4j;
import org.bytedeco.javacpp.opencv_core;
import org.bytedeco.javacv.FFmpegFrameGrabber;
import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.FrameGrabber;
import org.bytedeco.javacv.OpenCVFrameConverter;
import ramo.klevis.ml.tracking.yolo.Speed;
import ramo.klevis.ml.tracking.yolo.Yolo;

import java.io.File;
import java.io.IOException;
import java.util.concurrent.CountDownLatch;

import static org.bytedeco.javacpp.opencv_highgui.destroyAllWindows;
import static org.bytedeco.javacpp.opencv_highgui.waitKey;

@Slf4j
public class VideoPlayer {

    private final OpenCVFrameConverter.ToMat converter = new OpenCVFrameConverter.ToMat();
    private Yolo yolo = new Yolo();
    private final CountDownLatch countDownLatch;
    private volatile boolean stop = false;
    private Speed selectedIndex;

    public VideoPlayer() throws IOException {
        countDownLatch = new CountDownLatch(2);
    }

    public void startRealTimeVideoDetection(String videoFileName, String windowName,
                                            Speed selectedIndex,
                                            boolean realYolo,
                                            boolean outputFrames) throws Exception {
        this.selectedIndex = selectedIndex;
        log.info("Start detecting video " + videoFileName);
        log.info(windowName);
        yolo.initialize(selectedIndex, realYolo, windowName,outputFrames);
        startYoloThread(yolo, windowName);
        countDownLatch.countDown();
        runVideoMainThread(yolo, windowName, videoFileName, converter);

    }

    private void runVideoMainThread(Yolo yolo, String windowName, String videoFileName, OpenCVFrameConverter.ToMat toMat) throws Exception {
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

            Thread.sleep(60);
            opencv_core.Mat mat = toMat.convert(frame);
            opencv_core.Mat resizeMat = new opencv_core.Mat(selectedIndex.height, selectedIndex.width, mat.type());
            yolo.push(resizeMat, windowName);
            org.bytedeco.javacpp.opencv_imgproc.resize(mat, resizeMat, resizeMat.size());
//            yolo.predictBoundingBoxes(windowName);
            yolo.drawBoundingBoxesRectangles(frame, resizeMat, windowName);
            char key = (char) waitKey(20);
            // Exit this loop on escape:
            if (key == 27) {
                stop();
                break;
            }
        }
    }

    private FFmpegFrameGrabber initFrameGrabber(String videoFileName) throws FrameGrabber.Exception {
        FFmpegFrameGrabber grabber = new FFmpegFrameGrabber(new File(videoFileName));
        grabber.start();
        return grabber;
    }

    private void startYoloThread(Yolo yolo, String windowName) {
        Thread thread = new Thread(() -> {
            while (!stop) {
                try {
                    yolo.predictBoundingBoxes(windowName);
                } catch (Exception e) {
                    //ignoring a thread failure
                    //it may fail because the frame may be long gone when thread get chance to execute
                }
            }
            log.info("YOLO Thread Exit");
        });
        thread.start();
    }

    public void stop() throws IOException {
        if (!stop) {
            stop = true;
            yolo = new Yolo();
            destroyAllWindows();
        }
    }
}