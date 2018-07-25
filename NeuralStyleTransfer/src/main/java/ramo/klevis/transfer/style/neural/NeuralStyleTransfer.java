package ramo.klevis.transfer.style.neural;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.AdamUpdater;
import org.nd4j.linalg.learning.config.Adam;

import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * Neural Style Transfer Algorithm
 * References
 * https://arxiv.org/pdf/1508.06576.pdf
 * https://arxiv.org/pdf/1603.08155.pdf
 * https://harishnarayanan.org/writing/artistic-style-transfer/
 *
 * @author Klevis Ramo&Jacob Schrum
 */
@Slf4j
public class NeuralStyleTransfer {

    public static final String[] STYLE_LAYERS = new String[]{
            "block1_conv1,0.5",
            "block2_conv1,1.0",
            "block3_conv1,1.5",
            "block4_conv2,3.0",
            "block5_conv1,4.0"
    };
    private static final String CONTENT_LAYER_NAME = "block4_conv2";
    /**
     * ADAM
     * Possible values 0.8-0.95
     */
    private static final double BETA_MOMENTUM = 0.9;
    /**
     * ADAM
     * Below values rarely change
     */
    private static final double BETA2_MOMENTUM = 0.999;
    private static final double EPSILON = 0.00000008;

    /**
     * Values suggested by
     * https://harishnarayanan.org/writing/artistic-style-transfer/
     * Other Values(5,100): http://www.chioka.in/tensorflow-implementation-neural-algorithm-of-artistic-style
     */
    public static final double ALPHA = 5;
    public static final double BETA = 100;

    private static final double LEARNING_RATE = 2;
    private static final int ITERATIONS = 1000;
    /**
     * Higher resolution brings better results but
     * changing behind 300x400 on CPU become almost not computable and slow
     */
    public static final int HEIGHT = 224;
    public static final int WIDTH = 224;
    public static final int CHANNELS = 3;

    private ExecutorService executorService;

    final ImageUtilities imageUtilities = new ImageUtilities();
    private final ContentCostFunction contentCostFunction = new ContentCostFunction();
    private final StyleCostFunction styleCostFunction = new StyleCostFunction();

    public static void main(String[] args) throws Exception {
        new NeuralStyleTransfer().transferStyle();
    }

    public void transferStyle() throws Exception {

        ComputationGraph vgg16FineTune = ramo.klevis.transfer.style.neural.VGG16.loadModel();

        INDArray content = imageUtilities.loadImage(ImageUtilities.CONTENT_FILE);
        INDArray style = imageUtilities.loadImage(ImageUtilities.STYLE_FILE);
        INDArray generated = imageUtilities.createGeneratedImage();

        Map<String, INDArray> contentActivationsMap = vgg16FineTune.feedForward(content, true);
        Map<String, INDArray> styleActivationsMap = vgg16FineTune.feedForward(style, true);
        HashMap<String, INDArray> styleActivationsGramMap = buildStyleGramValues(styleActivationsMap);

        AdamUpdater adamUpdater = createADAMUpdater();
        executorService = Executors.newCachedThreadPool();

        for (int iteration = 0; iteration < ITERATIONS; iteration++) {
            long start = System.currentTimeMillis();
            log.info("iteration  " + iteration);

            CountDownLatch countDownLatch = new CountDownLatch(2);

            //activations of the generated image
            Map<String, INDArray> generatedActivationsMap = vgg16FineTune.feedForward(generated, true);

            final INDArray[] styleBackProb = new INDArray[1];
            executorService.execute(() -> {
                try {
                    styleBackProb[0] = backPropagateStyles(vgg16FineTune, styleActivationsGramMap, generatedActivationsMap);
                    countDownLatch.countDown();
                } catch (Exception e) {
                    throw new RuntimeException(e);
                }
            });
            final INDArray[] backPropContent = new INDArray[1];
            executorService.execute(() -> {
                backPropContent[0] = backPropagateContent(vgg16FineTune, contentActivationsMap, generatedActivationsMap);
                countDownLatch.countDown();
            });

            countDownLatch.await();
            INDArray backPropAllValues = backPropContent[0].muli(ALPHA).addi(styleBackProb[0].muli(BETA));
            adamUpdater.applyUpdater(backPropAllValues, iteration);
            generated.subi(backPropAllValues.get());

            System.out.println((System.currentTimeMillis()) - start);
//            log.info("Total Loss: " + totalLoss(styleActivationsMap, generatedActivationsMap, contentActivationsMap));
            if (iteration % ImageUtilities.SAVE_IMAGE_CHECKPOINT == 0) {
                //save image can be found at target/classes/styletransfer/out
                imageUtilities.saveImage(generated.dup(), iteration);
            }
        }

    }

    private INDArray backPropagateStyles(ComputationGraph vgg16FineTune,
                                         HashMap<String, INDArray> StyleActivationsGramMap,
                                         Map<String, INDArray> generatedActivationsMap) throws Exception {
        INDArray styleBackProb = Nd4j.zeros(new int[]{1, CHANNELS, HEIGHT, WIDTH});
        CountDownLatch countDownLatch = new CountDownLatch(STYLE_LAYERS.length);

        for (String styleLayer : STYLE_LAYERS) {
            String[] split = styleLayer.split(",");
            String styleLayerName = split[0];
            INDArray styleGramValues = StyleActivationsGramMap.get(styleLayerName);
            INDArray generatedValues = generatedActivationsMap.get(styleLayerName);
            double weight = Double.parseDouble(split[1]);
            int index = findLayerIndex(styleLayerName);
            executorService.execute(() -> {
                INDArray dStyleValues = styleCostFunction.styleContentFunctionDerivative(styleGramValues, generatedValues).transpose();
                INDArray backProb = backPropagate(vgg16FineTune, dStyleValues.reshape(generatedValues.shape()), index).muli(weight);
                styleBackProb.addi(backProb);
                countDownLatch.countDown();
            });
        }
        countDownLatch.await();
        return styleBackProb;
    }

    private INDArray backPropagateContent(ComputationGraph vgg16FineTune, Map<String, INDArray> contentActivationsMap, Map<String, INDArray> generatedActivationsMap) {
        INDArray contentActivationsAtLayer = contentActivationsMap.get(CONTENT_LAYER_NAME);
        INDArray generatedActivationsAtLayer = generatedActivationsMap.get(CONTENT_LAYER_NAME);
        INDArray dContentLayer = contentCostFunction.contentFunctionDerivative(contentActivationsAtLayer, generatedActivationsAtLayer);
        return backPropagate(vgg16FineTune, dContentLayer.reshape(generatedActivationsAtLayer.shape()), findLayerIndex(CONTENT_LAYER_NAME));
    }

    private AdamUpdater createADAMUpdater() {
        AdamUpdater adamUpdater = new AdamUpdater(new Adam(LEARNING_RATE, BETA_MOMENTUM, BETA2_MOMENTUM, EPSILON));
        adamUpdater.setStateViewArray(Nd4j.zeros(1, 2 * CHANNELS * WIDTH * HEIGHT),
                new int[]{1, CHANNELS, HEIGHT, WIDTH}, 'c',
                true);
        return adamUpdater;
    }

    /**
     * Since style activation are not changing we are saving
     * some computation by calculating style grams only once
     *
     * @param styleActivations
     * @return
     */
    private HashMap<String, INDArray> buildStyleGramValues(Map<String, INDArray> styleActivations) {
        HashMap<String, INDArray> styleGramValuesMap = new HashMap<>();
        for (String styleLayer : STYLE_LAYERS) {
            String[] split = styleLayer.split(",");
            String styleLayerName = split[0];
            INDArray styleValues = styleActivations.get(styleLayerName);
            styleGramValuesMap.put(styleLayerName, styleCostFunction.gramMatrix(styleValues));
        }
        return styleGramValuesMap;
    }

    private double totalLoss(Map<String, INDArray> styleActivationsMap, Map<String, INDArray> generatedActivationsMap, Map<String, INDArray> contentActivationsMap) {
        Double stylesLoss = styleCostFunction.allStylesContentFunction(styleActivationsMap, generatedActivationsMap);
        return ALPHA * contentCostFunction.contentFunction(generatedActivationsMap.get(CONTENT_LAYER_NAME).dup(), contentActivationsMap.get(CONTENT_LAYER_NAME).dup()) + BETA * stylesLoss;
    }

    private int findLayerIndex(String styleLayerName) {
        int index = 0;
        for (int i = 0; i < ramo.klevis.transfer.style.neural.VGG16.ALL_LAYERS.length; i++) {
            if (styleLayerName.equalsIgnoreCase(ramo.klevis.transfer.style.neural.VGG16.ALL_LAYERS[i])) {
                index = i;
                break;
            }
        }
        return index;
    }

    private INDArray backPropagate(ComputationGraph vgg16FineTune, INDArray dLdANext, int startFrom) {

        for (int i = startFrom; i > 0; i--) {
            Layer layer = vgg16FineTune.getLayer(ramo.klevis.transfer.style.neural.VGG16.ALL_LAYERS[i]);
            dLdANext = layer.backpropGradient(dLdANext).getSecond();
        }
        return dLdANext;
    }

}
