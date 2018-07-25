package ramo.klevis.ml;

import lombok.Getter;

import java.io.Serializable;

/**
 * Created by Klevis Ramo
 */
@Getter
public class LabeledImage implements Serializable {
    private final double[] normalizedPixels;
    private final double[] pixels;
    private final double label;

    public LabeledImage(int label, double[] pixels) {
        normalizedPixels = normalizePixels(pixels);
        this.pixels = pixels;
        this.label = label;
    }

    /**
     * Usually in computer vision dividing by 255 is a good normalization
     * or for VGG-16 maybe a mean subtraction is also fine
     * @param pixels
     * @return
     */
    private double[] normalizePixels(double[] pixels) {
        double[] pixelsNorm = new double[pixels.length];
        for (int i = 0; i < pixels.length; i++) {
            pixelsNorm[i] = pixels[i] / 255d;
        }
        return pixelsNorm;
    }

    @Override
    public String toString() {
        return "LabeledImage{" +
                "label=" + label +
                '}';
    }
}
