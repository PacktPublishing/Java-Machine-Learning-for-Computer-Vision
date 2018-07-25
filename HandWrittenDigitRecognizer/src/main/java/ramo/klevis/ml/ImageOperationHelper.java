package ramo.klevis.ml;

import com.mortennobel.imagescaling.ResampleFilters;
import com.mortennobel.imagescaling.ResampleOp;

import java.awt.*;
import java.awt.image.BufferedImage;

public class ImageOperationHelper {
    private static final int WIDTH = 28;
    private static final int HEIGHT = 28;

    public static BufferedImage scaleImage(BufferedImage imageToScale) {
        ResampleOp resizeOp = new ResampleOp(WIDTH, HEIGHT);
        resizeOp.setFilter(ResampleFilters.getLanczos3Filter());
        return resizeOp.filter(imageToScale, null);
    }

    public static BufferedImage toBufferedImage(Image img) {
        // Create a buffered image with transparency
        BufferedImage bimage = new BufferedImage(img.getWidth(null), img.getHeight(null), BufferedImage.TYPE_INT_ARGB);

        // Draw the image on to the buffered image
        Graphics2D bGr = bimage.createGraphics();
        bGr.drawImage(img, 0, 0, null);
        bGr.dispose();
        return bimage;
    }

    /**
     * Transforms a three dimensional image matrices into
     * a gray scale one dimensional vector
     *
     * @param img
     * @return
     */
    public static double[] transformRGBToGrayScale(BufferedImage img) {

        double[] imageGray = new double[28 * 28];
        int w = img.getWidth();
        int h = img.getHeight();
        int index = 0;
        for (int i = 0; i < w; i++) {
            for (int j = 0; j < h; j++) {
                Color color = new Color(img.getRGB(j, i), true);
                int red = (color.getRed());
                int green = (color.getGreen());
                int blue = (color.getBlue());
                //RGB-> black and white or gray scale
                double v = 255 - (red + green + blue) / 3d;
                imageGray[index] = v;
                index++;
            }
        }
        return imageGray;
    }
}
