package ramo.klevis.transfer.style.neural;

import org.nd4j.linalg.api.ndarray.INDArray;

import static ramo.klevis.transfer.style.neural.MatricesOperations.*;

public class ContentCostFunction {

    /**
     * Equation (2) from the Gatys et all paper: https://arxiv.org/pdf/1508.06576.pdf
     * This is the derivative of the content loss w.r.t. the combo image features
     * within a specific layer of the CNN.
     *
     * @param contentActivations Features at particular layer from the original content image
     * @param generatedActivations    Features at same layer from current combo image
     * @return Derivatives of content loss w.r.t. combo features
     */
    public INDArray contentFunctionDerivative(INDArray contentActivations, INDArray generatedActivations) {

        generatedActivations = generatedActivations.dup();
        contentActivations = contentActivations.dup();

        double channels = generatedActivations.shape()[0];
        double w = generatedActivations.shape()[1];
        double h = generatedActivations.shape()[2];

        double contentWeight = 1.0 / (2 * (channels) * (w) * (h));
        // Compute the F^l - P^l portion of equation (2), where F^l = comboFeatures and P^l = originalFeatures
        INDArray diff = generatedActivations.sub(contentActivations);
        // This multiplication assures that the result is 0 when the value from F^l < 0, but is still F^l - P^l otherwise
        return flatten(diff.muli(contentWeight).muli(ensurePositive(generatedActivations)));
    }


    /**
     * After passing in the content, style, and combination images,
     * compute the loss with respect to the content. Based off of:
     * https://harishnarayanan.org/writing/artistic-style-transfer/
     *
     * @param generatedActivations    Intermediate layer activations from the three inputs
     * @param contentActivations Intermediate layer activations from the three inputs
     * @return Weighted content loss component
     */

    public double contentFunction(INDArray generatedActivations, INDArray contentActivations) {
        int[] shape = generatedActivations.shape();
        int N = shape[0];
        int M = shape[1] * shape[2];
        return sumOfSquaredErrors(contentActivations, generatedActivations) / (4.0 * (N) * (M));
    }

}
