package ramo.klevis.transfer.style.neural;

import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.zoo.PretrainedType;
import org.deeplearning4j.zoo.ZooModel;

import java.io.IOException;

public class VGG16 {
    static final String[] ALL_LAYERS = new String[]{
            "input_1",
            "block1_conv1",
            "block1_conv2",
            "block1_pool",
            "block2_conv1",
            "block2_conv2",
            "block2_pool",
            "block3_conv1",
            "block3_conv2",
            "block3_conv3",
            "block3_pool",
            "block4_conv1",
            "block4_conv2",
            "block4_conv3",
            "block4_pool",
            "block5_conv1",
            "block5_conv2",
            "block5_conv3",
            "block5_pool",
            "flatten",
            "fc1",
            "fc2"
    };

    public static ComputationGraph loadModel() throws IOException {
        ZooModel zooModel = new org.deeplearning4j.zoo.model.VGG16();
        ComputationGraph vgg16 = (ComputationGraph) zooModel.initPretrained(PretrainedType.IMAGENET);
        ComputationGraph vgg16Transfer = new TransferLearning.GraphBuilder(vgg16)
                .removeVertexKeepConnections("block5_pool")
                .addLayer("block5_pool", avgPoolLayer(), "block5_conv3")
                .removeVertexKeepConnections("block4_pool")
                .addLayer("block4_pool", avgPoolLayer(), "block4_conv3")
                .removeVertexKeepConnections("block3_pool")
                .addLayer("block3_pool", avgPoolLayer(), "block3_conv3")
                .removeVertexKeepConnections("block2_pool")
                .addLayer("block2_pool", avgPoolLayer(), "block2_conv2")
                .removeVertexKeepConnections("block1_pool")
                .addLayer("block1_pool", avgPoolLayer(), "block1_conv2")
                .setInputTypes(InputType
                        .convolutionalFlat(NeuralStyleTransfer.HEIGHT, NeuralStyleTransfer.WIDTH, NeuralStyleTransfer.CHANNELS))
                .removeVertexAndConnections("fc2")
                .removeVertexAndConnections("fc1")
                .removeVertexAndConnections("flatten")
                .removeVertexAndConnections("predictions")
                .setOutputs("block5_pool")
                .build();
        vgg16Transfer.initGradientsView();
        System.out.println("vgg16Transfer.summary() = " + vgg16Transfer.summary());
        return vgg16Transfer;
    }

    private static SubsamplingLayer avgPoolLayer() {
        return new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.AVG)
                .kernelSize(2, 2)
                .stride(2, 2)
                .convolutionMode(ConvolutionMode.Same)
                .build();
    }
}
