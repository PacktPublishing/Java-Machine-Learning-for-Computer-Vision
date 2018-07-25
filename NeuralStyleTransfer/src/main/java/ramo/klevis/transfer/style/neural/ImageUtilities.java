package ramo.klevis.transfer.style.neural;

import org.datavec.image.loader.NativeImageLoader;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.VGG16ImagePreProcessor;
import org.nd4j.linalg.factory.Nd4j;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.concurrent.ThreadLocalRandom;

import static ramo.klevis.transfer.style.neural.NeuralStyleTransfer.*;

public class ImageUtilities {

    public static final String OUTPUT_PATH = "NeuralStyleTransfer/src/main/resources/generatedImages";
    public static final DataNormalization IMAGE_PRE_PROCESSOR = new VGG16ImagePreProcessor();
    public static final String CONTENT_FILE = "NeuralStyleTransfer/src/main/resources/content.jpg";
    public static final String STYLE_FILE = "NeuralStyleTransfer/src/main/resources/style.jpg";
    public static final int SAVE_IMAGE_CHECKPOINT = 5;
    public static final NativeImageLoader LOADER = new NativeImageLoader(HEIGHT, WIDTH, CHANNELS);
    public static final double NOISE_RATION = 0.2;

    public BufferedImage saveImage(INDArray combination, int iteration) throws IOException {
        IMAGE_PRE_PROCESSOR.revertFeatures(combination);

        BufferedImage output = imageFromINDArray(combination);
        File file = new File(OUTPUT_PATH + "/iteration" + iteration + ".jpg");
        ImageIO.write(output, "jpg", file);
        return output;
    }

    /**
     * Takes an INDArray containing an image loaded using the native image loader
     * libraries associated with DL4J, and converts it into a BufferedImage.
     * The INDArray contains the color values split up across three channels (RGB)
     * and in the integer range 0-255.
     *
     * @param array INDArray containing an image
     * @return BufferedImage
     */
    public BufferedImage imageFromINDArray(INDArray array) {
        int[] shape = array.shape();

        int height = shape[2];
        int width = shape[3];
        BufferedImage image = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        for (int x = 0; x < width; x++) {
            for (int y = 0; y < height; y++) {
                int red = array.getInt(0, 2, y, x);
                int green = array.getInt(0, 1, y, x);
                int blue = array.getInt(0, 0, y, x);

                //handle out of bounds pixel values
                red = Math.min(red, 255);
                green = Math.min(green, 255);
                blue = Math.min(blue, 255);

                red = Math.max(red, 0);
                green = Math.max(green, 0);
                blue = Math.max(blue, 0);
                image.setRGB(x, y, new Color(red, green, blue).getRGB());
            }
        }
        return image;
    }

    public INDArray createGeneratedImage() throws IOException {
        INDArray content = LOADER.asMatrix(new File(CONTENT_FILE));
        IMAGE_PRE_PROCESSOR.transform(content);
        INDArray combination = createCombineImageWithRandomPixels();
        combination.muli(NOISE_RATION).addi(content.muli(1.0 - NOISE_RATION));
        return combination;
    }

    public INDArray createCombineImageWithRandomPixels() {
        int totalEntries = CHANNELS * HEIGHT * WIDTH;
        double[] result = new double[totalEntries];
        for (int i = 0; i < result.length; i++) {
            result[i] = ThreadLocalRandom.current().nextDouble(-20, 20);
        }
        return Nd4j.create(result, new int[]{1, CHANNELS, HEIGHT, WIDTH});
    }

    public INDArray loadImage(String contentFile) throws IOException {
        INDArray content = LOADER.asMatrix(new File(contentFile));
        //scaling and image normalization
        IMAGE_PRE_PROCESSOR.transform(content);
        return content;
    }
}
