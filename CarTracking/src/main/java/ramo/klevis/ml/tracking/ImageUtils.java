package ramo.klevis.ml.tracking;

import org.bytedeco.javacpp.opencv_core;
import org.bytedeco.javacpp.opencv_imgcodecs;
import org.deeplearning4j.nn.layers.objdetect.DetectedObject;
import ramo.klevis.ml.tracking.yolo.Speed;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.File;
import java.nio.ByteBuffer;

/**
 * Created by klevis.ramo on 11/5/2018
 */

public class ImageUtils {

    public static BufferedImage mat2BufferedImage(opencv_core.Mat matrix) throws Exception {
        ByteBuffer allocate = ByteBuffer.allocate(matrix.arraySize());
        opencv_imgcodecs.imencode(".jpg", matrix, allocate);
        byte ba[] = allocate.array();
        BufferedImage bi = ImageIO.read(new ByteArrayInputStream(ba));
        return bi;
    }

    public static BufferedImage cropImageWithYolo(Speed speed,
                                                  opencv_core.Mat mat,
                                                  DetectedObject obj,
                                                  boolean writeCropImageIntoDisk) throws Exception {
        double wPixelPerGrid = speed.width / speed.gridWidth;
        double hPixelPerGrid = speed.height / speed.gridHeight;
        double tx = Math.abs(obj.getTopLeftXY()[0] * wPixelPerGrid);
        double ty = Math.abs(obj.getTopLeftXY()[1] * hPixelPerGrid);

        BufferedImage image = mat2BufferedImage(mat);
        double width = obj.getWidth();
        double height = obj.getHeight();
        if (((width * wPixelPerGrid) + tx) > speed.width) {
            width = (speed.width - tx) / wPixelPerGrid;
        }
        if (((height * hPixelPerGrid) + ty) > speed.height) {
            height = (speed.height - ty) / hPixelPerGrid;
        }
        BufferedImage subimage = image.getSubimage((int) tx, (int) ty, (int) (width * wPixelPerGrid), (int) (height * hPixelPerGrid));
        if (writeCropImageIntoDisk) {
            ImageIO.write(subimage, "jpg", new File("AutonomousDriving/src/main/resources/videoFrames/"+ System.currentTimeMillis() + ".jpg"));
        }
        return subimage;
    }
}
