package ramo.klevis.transfer.style.neural;

import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Map;

import static ramo.klevis.transfer.style.neural.MatricesOperations.*;
import static ramo.klevis.transfer.style.neural.NeuralStyleTransfer.STYLE_LAYERS;

public class StyleCostFunction {
    /**
     * This method is simply called style_loss in
     * https://harishnarayanan.org/writing/artistic-style-transfer/
     * but it takes inputs for intermediate activations from a particular
     * layer, hence my re-name. These values contribute to the total
     * style loss.
     *
     * @param style       Activations from intermediate layer of CNN for style image input
     * @param generated Activations from intermediate layer of CNN for combination image input
     * @return Loss contribution from this comparison
     */
    public double styleContentFunction(INDArray style, INDArray generated) {
        INDArray s = gramMatrix(style);
        INDArray c = gramMatrix(generated);
        int[] shape = style.shape();
        int N = shape[0];
        int M = shape[1] * shape[2];
        return sumOfSquaredErrors(s, c) / (4.0 * (N * N) * (M * M));
    }


    public Double allStylesContentFunction(Map<String, INDArray> stykeActivationsMap, Map<String, INDArray> generatedActivationsMap) {
        Double styles = 0.0;
        for (String styleLayers : STYLE_LAYERS) {
            String[] split = styleLayers.split(",");
            String styleLayerName = split[0];
            double weight = Double.parseDouble(split[1]);
            styles += styleContentFunction(stykeActivationsMap.get(styleLayerName).dup(), generatedActivationsMap.get(styleLayerName).dup()) * weight;
        }
        return styles;
    }


    /**
     * Equation (6) from the Gatys et all paper: https://arxiv.org/pdf/1508.06576.pdf
     * This is the derivative of the style error for a single layer w.r.t. the
     * combo image features at that layer.
     *
     * @param styleGramFeatures Intermediate activations of one layer for style input
     * @param generatedFeatures     Intermediate activations of one layer for combo image input
     * @return Derivative of style error matrix for the layer w.r.t. combo image
     */
    public INDArray styleContentFunctionDerivative(INDArray styleGramFeatures, INDArray generatedFeatures) {

        generatedFeatures = generatedFeatures.dup();
        double N = generatedFeatures.shape()[0];
        double M = generatedFeatures.shape()[1] * generatedFeatures.shape()[2];

        double styleWeight = 1.0 / ((N * N) * (M * M));
        // Corresponds to G^l in equation (6)
        INDArray contentGram = gramMatrix(generatedFeatures);
        // G^l - A^l
        INDArray diff = contentGram.sub(styleGramFeatures);
        // (F^l)^T * (G^l - A^l)
        INDArray trans = flatten(generatedFeatures).transpose();
        INDArray product = trans.mmul(diff);
        // (1/(N^2 * M^2)) * ((F^l)^T * (G^l - A^l))
        INDArray posResult = product.muli(styleWeight);
        // This multiplication assures that the result is 0 when the value from F^l < 0, but is still (1/(N^2 * M^2)) * ((F^l)^T * (G^l - A^l)) otherwise
        return posResult.muli(ensurePositive(trans));
    }


    /**
     * Computing the Gram matrix as described here:
     * https://harishnarayanan.org/writing/artistic-style-transfer/
     * Permuting dimensions is not needed because DL4J stores
     * the channel at the front rather than the end of the tensor.
     * Basically, each tensor is flattened into a vector so that
     * the dot product can be calculated.
     *
     * @param x Tensor to get Gram matrix of
     * @return Resulting Gram matrix
     */
    public INDArray gramMatrix(INDArray x) {
        INDArray flattened = flatten(x);
        return flattened.mmul(flattened.transpose());
    }
}
