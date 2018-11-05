package ramo.klevis.ml.tracking.data;

import org.bytedeco.javacpp.opencv_core;
import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.layers.objdetect.DetectedObject;
import org.deeplearning4j.nn.layers.objdetect.Yolo2OutputLayer;
import org.deeplearning4j.nn.layers.objdetect.YoloUtils;
import org.deeplearning4j.zoo.model.YOLO2;
import org.jetbrains.annotations.NotNull;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import ramo.klevis.ml.tracking.ImageUtils;
import ramo.klevis.ml.tracking.yolo.Speed;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Optional;
import java.util.concurrent.ThreadLocalRandom;
import java.util.stream.Stream;

/**
 * Created by klevis.ramo on 10/14/2018
 */

public class TransformRotatingImages {

    private static final String BASE_PATH = "CarTracking\\src\\main\\resources\\tripod-seq";
    private static final ImagePreProcessingScaler PRE_PROCESSOR = new ImagePreProcessingScaler(0, 1);
    private static final String TRANSFORMED = "\\transformed";
    private static NativeImageLoader LOADER;

    public static void main(String[] args) throws Exception {
        takeOneForEachThree();
        cropAllWithYOLOBoundingBox();
        moveImageToTheLeft();
        resizeSomeOfTheImages();
        moveFilesToFolders();
        cleanUp();
    }

    private static void takeOneForEachThree() {
        File file = new File(BASE_PATH);
        File[] files = file.listFiles();
        sortFiles(files);
        List<File> toBeDeleted = new ArrayList<>();
        int i = 0;
        String prevClassNumber = null;
        for (File file1 : files) {
            //tripod_seq_01_002.jpg
            if (file1.isDirectory()) {
                continue;
            }

            String name = file1.getName();
            String number = name.substring(name.indexOf("seq_") + 4,
                    name.indexOf(".jpg"));
            String classNumber = number.substring(0, number.indexOf("_"));
            if (i % 3 != 0) {
                toBeDeleted.add(file1);
            }

            if (prevClassNumber != null && !prevClassNumber.equals(classNumber)) {
                i = 0;
            }

            prevClassNumber = classNumber;
            i++;
        }
        toBeDeleted.stream().parallel().forEach(e -> e.delete());
    }

    private static void cleanUp() throws IOException {
        File file = new File(BASE_PATH + TRANSFORMED);
        String[] list = file.list();
        int count = 0;
        System.out.println("list.length = " + list.length);
        for (String fileName : list) {
            String pathname = BASE_PATH + TRANSFORMED + "\\" + fileName;
            String[] list1 = new File(pathname).list();
            if (list1 == null) {
                continue;
            }
            System.out.println(list1.length + " " + pathname);
            for (String s : list1) {
                File input = new File(pathname + "\\" + s);
                BufferedImage bufferedImage = ImageIO.read(input);
                if (bufferedImage == null) {
                    System.out.println("No Image " + input.getAbsolutePath());
                    input.delete();
                    continue;
                }
                if (bufferedImage.getWidth() <= 10 || bufferedImage.getHeight() <= 10) {
                    count++;
                    input.delete();
                }
            }
        }
        System.out.println("Cleaned  " + count);
    }

    private static void moveFilesToFolders() {
        File file = new File(BASE_PATH + TRANSFORMED);
        String[] list = file.list();
        for (String fileName : list) {
            String classFolder = fileName.substring(fileName.indexOf("q_") + 2, fileName.indexOf("q_") + 4);
            System.out.println("substring = " + classFolder);
            File folder = new File(BASE_PATH + "\\" + TRANSFORMED + "\\" + classFolder);
            if (!folder.exists()) {
                folder.mkdir();
            }
            new File(BASE_PATH + TRANSFORMED + "\\" + fileName)
                    .renameTo(new File(BASE_PATH + TRANSFORMED + "\\" + classFolder + "\\" + fileName));
        }
    }

    private static void resizeSomeOfTheImages() throws IOException {
        File file = new File(BASE_PATH + TRANSFORMED);
        String[] list = file.list();
        List<String> imagesNames = Arrays.asList(list);
        Collections.shuffle(imagesNames);
        //randomly choose 30% of images for rescale
        int size = (int) (imagesNames.size() * 0.4);
        List<String> strings = imagesNames.subList(0, size);
        strings.stream().parallel().forEach(fileName -> {
            String pathname = BASE_PATH + TRANSFORMED + "\\" + fileName;
            if (new File(pathname).isDirectory()) {
                return;
            }
            try {
                File input = new File(pathname);
                BufferedImage bufferedImage = ImageIO.read(input);
                double w = ThreadLocalRandom.current().nextInt(25, 40) / 100.d;
                double h = ThreadLocalRandom.current().nextInt(25, 40) / 100.d;

                BufferedImage scaledInstance = resizeImage(bufferedImage, (int) (bufferedImage.getWidth() * w),
                        (int) (bufferedImage.getHeight() * h), bufferedImage.getType());
                File output = new File(BASE_PATH + TRANSFORMED + "\\" + fileName);

                ImageIO.write(scaledInstance, "jpg", output);
                System.out.println("Resized " + output.getAbsolutePath());
            } catch (IOException e) {
                e.printStackTrace();
            }

        });


    }

    private static BufferedImage resizeImage(BufferedImage originalImage, int width, int height, int type) {
        BufferedImage resizedImage = new BufferedImage(width, height, type);
        Graphics2D g = resizedImage.createGraphics();
        g.drawImage(originalImage, 0, 0, width, height, null);
        g.dispose();

        return resizedImage;
    }

    private static void moveImageToTheLeft() throws IOException {
        File file = new File(BASE_PATH + TRANSFORMED);
        String[] list = file.list();
        for (String fileName : list) {
            String pathname = BASE_PATH + TRANSFORMED + "\\" + fileName;
            if (new File(pathname).isDirectory()) {
                continue;
            }
            BufferedImage bufferedImage = ImageIO.read(new File(pathname));
            int from = 0;
            for (int i = 0; i < 3; i++) {
                double percentage = ThreadLocalRandom.current().nextInt(20, 30) / 100.0;
                int x = (int) (percentage * bufferedImage.getWidth());
                from = from + x;
                BufferedImage subimage = bufferedImage.getSubimage(from, 0, bufferedImage.getWidth() - from, bufferedImage.getHeight());
                ImageIO.write(subimage, "jpg", new File(BASE_PATH + TRANSFORMED + "\\" + fileName.replace(".jpg", "h_" + i + ".jpg")));
            }
        }
    }

    private static void cropAllWithYOLOBoundingBox() throws Exception {
        ComputationGraph yolo = (ComputationGraph) YOLO2.builder().build().initPretrained();
        File file = new File(BASE_PATH);
        File[] list = file.listFiles();
        sortFiles(list);
        Speed selectedSpeed = Speed.MEDIUM;
        LOADER = new NativeImageLoader(selectedSpeed.height, selectedSpeed.width, 3);
        Stream.of(list).parallel().forEach(e -> {
            try {
                cropImageWithYOLOBoundingBox(yolo, selectedSpeed, e);
            } catch (Exception e1) {
                e1.printStackTrace();
            }
        });
    }

    private static void sortFiles(File[] list) {
        Arrays.sort(list, new Comparator<File>() {
            @Override
            public int compare(File o1, File o2) {
                if (o1.isDirectory() || o2.isDirectory()) {
                    return 1;
                }

                String number = extractNumberFromName(o1.getName());
                String number2 = extractNumberFromName(o2.getName());
                return Integer.valueOf(number).compareTo(Integer.valueOf(number2));
            }

            @NotNull
            private String extractNumberFromName(String name) {
                //tripod_seq_01_002.jpg
                return name.substring(name.indexOf("seq_") + 4, name.indexOf(".jpg"))
                        .replace("_", "");
            }
        });
    }

    private static void cropImageWithYOLOBoundingBox(ComputationGraph yolo,
                                                     Speed selectedSpeed, File file) throws Exception {
        if (file.isDirectory()) {
            return;
        }
        BufferedImage bufferedImage = ImageIO.read(file);
        INDArray features = LOADER.asMatrix(bufferedImage);
        opencv_core.Mat mat = LOADER.asMat(features);
        PRE_PROCESSOR.transform(features);
        INDArray results = yolo.outputSingle(features);
        Yolo2OutputLayer outputLayer = (Yolo2OutputLayer) yolo.getOutputLayer(0);
        List<DetectedObject> predictedObjects = outputLayer.getPredictedObjects(results, 0.5);
        YoloUtils.nms(predictedObjects, 0.5);
        Optional<DetectedObject> max = predictedObjects.stream()
                .max((o1, o2) -> ((Double) o1.getConfidence()).compareTo(o2.getConfidence()));
        createCroppedImage(mat, selectedSpeed, max.get(), file);
    }

    public static void createCroppedImage(opencv_core.Mat mat, Speed speed, DetectedObject obj, File file) throws Exception {
        double wPixelPerGrid = speed.width / speed.gridWidth;
        double hPixelPerGrid = speed.height / speed.gridHeight;
        double tx = Math.abs(obj.getTopLeftXY()[0] * wPixelPerGrid);
        double ty = Math.abs(obj.getTopLeftXY()[1] * hPixelPerGrid);
        BufferedImage image = ImageUtils.mat2BufferedImage(mat);
        double width = obj.getWidth();
        double height = obj.getHeight();
        if (((width * wPixelPerGrid) + tx) > speed.width) {
            width = (speed.width - tx) / wPixelPerGrid;
        }
        if (((height * hPixelPerGrid) + ty) > speed.height) {
            height = (speed.height - ty) / hPixelPerGrid;
        }
        File folder = new File(BASE_PATH + TRANSFORMED);
        if (!folder.exists()) {
            folder.mkdir();
        }
        BufferedImage subimage = image.getSubimage((int) tx, (int) ty, (int) (width * wPixelPerGrid), (int) (height * hPixelPerGrid));
        ImageIO.write(subimage, "jpg", new File(BASE_PATH + "\\transformed\\"
                + file.getName()));
    }
}
