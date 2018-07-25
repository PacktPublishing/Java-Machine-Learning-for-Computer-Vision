package ramo.klevis.ml.recogntion.face;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.graph.L2NormalizeVertex;
import org.deeplearning4j.nn.conf.graph.MergeVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.conf.layers.ZeroPaddingLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.learning.config.IUpdater;

import java.io.IOException;

import static ramo.klevis.ml.recogntion.face.FaceNetSmallV2Helper.*;

/**
 * Created by Klevis Ramo
 * <p>
 * A variant of the original FaceNetSmallV2Model model that relies on encodings and triplet loss
 * <p>
 * Inspired by keras implementation https://github.com/iwantooxxoox/Keras-OpenFace
 */
public class FaceNetSmallV2Model {

    private int numClasses = 0;
    private final long seed = 1234;
    private int[] inputShape = new int[]{3, 96, 96};
    private IUpdater updater = new Adam(0.1, 0.9, 0.999, 0.01);
    private int encodings = 128;
    public static int reluIndex = 1;
    public static int paddingIndex = 1;

    public ComputationGraphConfiguration conf() {

        ComputationGraphConfiguration.GraphBuilder graph = new NeuralNetConfiguration.Builder().seed(seed)
                .activation(Activation.IDENTITY)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(updater)
                .weightInit(WeightInit.RELU)
                .l2(5e-5)
                .miniBatch(true)
                .graphBuilder();


        graph.addInputs("input1")
                .addLayer("pad1",
                        zeroPadding(3), "input1")
                .addLayer("conv1",
                        convolution(7, inputShape[0], 64, 2),
                        "pad1")
                .addLayer("bn1", batchNorm(64),
                        "conv1")
                .addLayer(nextReluId(), relu(),
                        "bn1")
                .addLayer("pad2",
                        zeroPadding(1), lastReluId())
                // pool -> norm
                .addLayer("pool1",
                        new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[]{3, 3},
                                new int[]{2, 2})
                                .convolutionMode(ConvolutionMode.Truncate)
                                .build(),
                        "pad2")

                // Inception 2
                .addLayer("conv2",
                        convolution(1, 64, 64),
                        "pool1")
                .addLayer("bn2", batchNorm(64),
                        "conv2")
                .addLayer(nextReluId(),
                        relu(),
                        "bn2")

                .addLayer("pad3",
                        zeroPadding(1), lastReluId())

                .addLayer("conv3",
                        convolution(3, 64, 192),
                        "pad3")
                .addLayer("bn3",
                        batchNorm(192),
                        "conv3")
                .addLayer(nextReluId(),
                        relu(),
                        "bn3")

                .addLayer("pad4",
                        zeroPadding(1), lastReluId())
                .addLayer("pool2",
                        new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[]{3, 3},
                                new int[]{2, 2})
                                .convolutionMode(ConvolutionMode.Truncate)
                                .build(),
                        "pad4");


        buildBlock3a(graph);
        buildBlock3b(graph);
        buildBlock3c(graph);

        buildBlock4a(graph);
        buildBlock4e(graph);

        buildBlock5a(graph);
        buildBlock5b(graph);

        graph.addLayer("avgpool",
                new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.AVG, new int[]{3, 3},
                        new int[]{1, 1})
                        .convolutionMode(ConvolutionMode.Truncate)
                        .build(),
                "inception_5b")
                .addLayer("dense", new DenseLayer.Builder().nIn(736).nOut(encodings)
                        .activation(Activation.IDENTITY).build(), "avgpool")
                .addVertex("encodings", new L2NormalizeVertex(new int[]{}, 1e-12), "dense")
                .setInputTypes(InputType.convolutional(96, 96, inputShape[0])).pretrain(true);

       /* Uncomment in case of training the network, graph.setOutputs should be lossLayer then
        .addLayer("lossLayer", new CenterLossOutputLayer.Builder()
                        .lossFunction(LossFunctions.LossFunction.SQUARED_LOSS)
                        .activation(Activation.SOFTMAX).nIn(128).nOut(numClasses).lambda(1e-4).alpha(0.9)
                        .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer).build(),
                "embeddings")*/
        graph.setOutputs("encodings");

        return graph.build();
    }

    private void buildBlock3a(ComputationGraphConfiguration.GraphBuilder graph) {
        graph.addLayer("inception_3a_3x3_conv1", convolution(1, 192, 96),
                "pool2")
                .addLayer("inception_3a_3x3_bn1",
                        batchNorm(96), "inception_3a_3x3_conv1")
                .addLayer(nextReluId(),
                        relu(), "inception_3a_3x3_bn1")
                .addLayer(nextPaddingId(),
                        zeroPadding(1), lastReluId())
                .addLayer("inception_3a_3x3_conv2", convolution(3, 96, 128), lastPaddingId())
                .addLayer("inception_3a_3x3_bn2",
                        batchNorm(128),
                        "inception_3a_3x3_conv2")
                .addLayer(nextReluId(),
                        relu(), "inception_3a_3x3_bn2")

                .addLayer("inception_3a_5x5_conv1", convolution(1, 192, 16),
                        "pool2")
                .addLayer("inception_3a_5x5_bn1",
                        batchNorm(16),
                        "inception_3a_5x5_conv1")
                .addLayer(nextReluId(),
                        relu(), "inception_3a_5x5_bn1")
                .addLayer(nextPaddingId(),
                        zeroPadding(2), lastReluId())
                .addLayer("inception_3a_5x5_conv2", convolution(5, 16, 32), lastPaddingId())
                .addLayer("inception_3a_5x5_bn2",
                        batchNorm(32),
                        "inception_3a_5x5_conv2")
                .addLayer(nextReluId(),
                        relu(), "inception_3a_5x5_bn2")

                .addLayer("pool3",
                        new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[]{3, 3},
                                new int[]{2, 2})
                                .convolutionMode(ConvolutionMode.Truncate)
                                .build(),
                        "pool2")
                .addLayer("inception_3a_pool_conv", convolution(1, 192, 32), "pool3")
                .addLayer("inception_3a_pool_bn",
                        batchNorm(32),
                        "inception_3a_pool_conv")
                .addLayer(nextReluId(),
                        relu(),
                        "inception_3a_pool_bn")

                .addLayer(nextPaddingId(),
                        new ZeroPaddingLayer.Builder(new int[]{3, 4, 3, 4})
                                .build(), lastReluId())

                .addLayer("inception_3a_1x1_conv", convolution(1, 192, 64),
                        "pool2")
                .addLayer("inception_3a_1x1_bn",
                        batchNorm(64),
                        "inception_3a_1x1_conv")
                .addLayer(nextReluId(),
                        relu(),
                        "inception_3a_1x1_bn")
                .addVertex("inception_3a", new MergeVertex(), "relu5", "relu7", lastPaddingId(), "relu9");

    }


    private void buildBlock3b(ComputationGraphConfiguration.GraphBuilder graph) {
        graph.addLayer("inception_3b_3x3_conv1",
                convolution(1, 256, 96),
                "inception_3a")

                .addLayer("inception_3b_3x3_bn1",
                        batchNorm(96),
                        "inception_3b_3x3_conv1")

                .addLayer(nextReluId(),
                        relu(),
                        "inception_3b_3x3_bn1")

                .addLayer(nextPaddingId(),
                        zeroPadding(1), lastReluId())

                .addLayer("inception_3b_3x3_conv2",
                        convolution(3, 96, 128),
                        lastPaddingId())

                .addLayer("inception_3b_3x3_bn2",
                        batchNorm(128),
                        "inception_3b_3x3_conv2")

                .addLayer(nextReluId(),
                        relu(),
                        "inception_3b_3x3_bn2");


        graph.addLayer("inception_3b_5x5_conv1",
                convolution(1, 256, 32),
                "inception_3a")

                .addLayer("inception_3b_5x5_bn1",
                        batchNorm(32),
                        "inception_3b_5x5_conv1")

                .addLayer(nextReluId(),
                        relu(),
                        "inception_3b_5x5_bn1")
                .addLayer(nextPaddingId(),
                        zeroPadding(2), lastReluId())

                .addLayer("inception_3b_5x5_conv2",
                        convolution(5, 32, 64),
                        lastPaddingId())

                .addLayer("inception_3b_5x5_bn2",
                        batchNorm(64),
                        "inception_3b_5x5_conv2")
                .addLayer(nextReluId(),
                        relu(),
                        "inception_3b_5x5_bn2");

        graph.addLayer("avg1",
                new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.AVG, new int[]{3, 3},
                        new int[]{3, 3})
                        .convolutionMode(ConvolutionMode.Truncate)
                        .build(),
                "inception_3a")
                .addLayer("inception_3b_pool_conv",
                        convolution(1, 256, 64),
                        "avg1")

                .addLayer("inception_3b_pool_bn",
                        batchNorm(64),
                        "inception_3b_pool_conv")

                .addLayer(nextReluId(),
                        relu(),
                        "inception_3b_pool_bn")
                .addLayer(nextPaddingId(),
                        zeroPadding(4), lastReluId())

                .addLayer("inception_3b_1x1_conv",
                        convolution(1, 256, 64),
                        "inception_3a")
                .addLayer("inception_3b_1x1_bn",
                        batchNorm(64),
                        "inception_3b_1x1_conv")

                .addLayer(nextReluId(),
                        relu(),
                        "inception_3b_1x1_bn")
                .addVertex("inception_3b", new MergeVertex(), "relu11", "relu13", lastPaddingId(), "relu15");

    }

    private void buildBlock3c(ComputationGraphConfiguration.GraphBuilder graph) {
        convolution2dAndBN(graph, "inception_3c_3x3",
                128, 320, new int[]{1, 1}, new int[]{1, 1},
                256, 128, new int[]{3, 3}, new int[]{2, 2},
                new int[]{1, 1, 1, 1}, "inception_3b");
        String rel1 = lastReluId();

        convolution2dAndBN(graph, "inception_3c_5x5",
                32, 320, new int[]{1, 1}, new int[]{1, 1},
                64, 32, new int[]{5, 5}, new int[]{2, 2},
                new int[]{2, 2, 2, 2}, "inception_3b");
        String rel2 = lastReluId();

        graph.addLayer("pool7",
                new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[]{3, 3},
                        new int[]{2, 2})
                        .convolutionMode(ConvolutionMode.Truncate)
                        .build(),
                "inception_3b");

        graph.addLayer(nextPaddingId(),
                new ZeroPaddingLayer.Builder(new int[]{0, 1, 0, 1})
                        .build(), "pool7");
        String pad1 = lastPaddingId();

        graph.addVertex("inception_3c", new MergeVertex(), rel1, rel2, pad1);
    }

    private void buildBlock4a(ComputationGraphConfiguration.GraphBuilder graph) {
        convolution2dAndBN(graph, "inception_4a_3x3",
                96, 640, new int[]{1, 1}, new int[]{1, 1},
                192, 96, new int[]{3, 3}, new int[]{1, 1}
                , new int[]{1, 1, 1, 1}, "inception_3c");
        String rel1 = lastReluId();

        convolution2dAndBN(graph, "inception_4a_5x5",
                32, 640, new int[]{1, 1}, new int[]{1, 1},
                64, 32, new int[]{5, 5}, new int[]{1, 1}
                , new int[]{2, 2, 2, 2}, "inception_3c");
        String rel2 = lastReluId();

        graph.addLayer("avg7",
                new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.AVG, new int[]{3, 3},
                        new int[]{3, 3})
                        .convolutionMode(ConvolutionMode.Truncate)
                        .build(),
                "inception_3c");
        convolution2dAndBN(graph, "inception_4a_pool",
                128, 640, new int[]{1, 1}, new int[]{1, 1},
                null, null, null, null
                , new int[]{2, 2, 2, 2}, "avg7");
        String pad1 = lastPaddingId();

        convolution2dAndBN(graph, "inception_4a_1x1",
                256, 640, new int[]{1, 1}, new int[]{1, 1},
                null, null, null, null
                , null, "inception_3c");
        String rel4 = lastReluId();
        graph.addVertex("inception_4a", new MergeVertex(), rel1, rel2, rel4, pad1);

    }

    private void buildBlock4e(ComputationGraphConfiguration.GraphBuilder graph) {
        convolution2dAndBN(graph, "inception_4e_3x3",
                160, 640, new int[]{1, 1}, new int[]{1, 1},
                256, 160, new int[]{3, 3}, new int[]{2, 2},
                new int[]{1, 1, 1, 1}, "inception_4a");
        String rel1 = lastReluId();

        convolution2dAndBN(graph, "inception_4e_5x5",
                64, 640, new int[]{1, 1}, new int[]{1, 1},
                128, 64, new int[]{5, 5}, new int[]{2, 2},
                new int[]{2, 2, 2, 2}, "inception_4a");
        String rel2 = lastReluId();

        graph.addLayer("pool8",
                new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[]{3, 3},
                        new int[]{2, 2})
                        .convolutionMode(ConvolutionMode.Truncate)
                        .build(),
                "inception_4a");
        graph.addLayer(nextPaddingId(),
                new ZeroPaddingLayer.Builder(new int[]{0, 1, 0, 1})
                        .build(), "pool8");
        String pad1 = lastPaddingId();

        graph.addVertex("inception_4e", new MergeVertex(), rel1, rel2, pad1);
    }

    private void buildBlock5a(ComputationGraphConfiguration.GraphBuilder graph) {
        convolution2dAndBN(graph, "inception_5a_3x3",
                96, 1024, new int[]{1, 1}, new int[]{1, 1},
                384, 96, new int[]{3, 3}, new int[]{1, 1},
                new int[]{1, 1, 1, 1}, "inception_4e");
        String relu1 = lastReluId();

        graph.addLayer("avg9",
                new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.AVG, new int[]{3, 3},
                        new int[]{3, 3})
                        .convolutionMode(ConvolutionMode.Truncate)
                        .build(),
                "inception_4e");
        convolution2dAndBN(graph, "inception_5a_pool",
                96, 1024, new int[]{1, 1}, new int[]{1, 1},
                null, null, null, null,
                new int[]{1, 1, 1, 1}, "avg9");
        String pad1 = lastPaddingId();

        convolution2dAndBN(graph, "inception_5a_1x1",
                256, 1024, new int[]{1, 1}, new int[]{1, 1},
                null, null, null, null,
                null, "inception_4e");
        String rel3 = lastReluId();

        graph.addVertex("inception_5a", new MergeVertex(), relu1, pad1, rel3);
    }

    private void buildBlock5b(ComputationGraphConfiguration.GraphBuilder graph) {
        convolution2dAndBN(graph, "inception_5b_3x3",
                96, 736, new int[]{1, 1}, new int[]{1, 1},
                384, 96, new int[]{3, 3}, new int[]{1, 1},
                new int[]{1, 1, 1, 1}, "inception_5a");
        String rel1 = lastReluId();

        graph.addLayer("max2",
                new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[]{3, 3},
                        new int[]{2, 2})
                        .convolutionMode(ConvolutionMode.Truncate)
                        .build(),
                "inception_5a");
        convolution2dAndBN(graph, "inception_5b_pool",
                96, 736, new int[]{1, 1}, new int[]{1, 1},
                null, null, null, null,
                null, "max2");
        graph.addLayer(nextPaddingId(),
                zeroPadding(1), lastReluId());
        String pad1 = lastPaddingId();

        convolution2dAndBN(graph, "inception_5b_1x1",
                256, 736, new int[]{1, 1}, new int[]{1, 1},
                null, null, null, null,
                null, "inception_5a");
        String rel2 = lastReluId();

        graph.addVertex("inception_5b", new MergeVertex(), rel1, pad1, rel2);
    }

    public ComputationGraph init() throws IOException {
        resetIndexes();
        ComputationGraph computationGraph = new ComputationGraph(conf());
        computationGraph.init();
        loadWeights(computationGraph);
        return computationGraph;
    }

    private static void resetIndexes() {
        reluIndex = 1;
        paddingIndex = 1;
    }
}
