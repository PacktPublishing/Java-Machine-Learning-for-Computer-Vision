package ramo.klevis.ml.recogntion.face;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.layers.ActivationLayer;
import org.deeplearning4j.nn.conf.layers.BatchNormalization;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.ZeroPaddingLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

import static ramo.klevis.ml.recogntion.face.FaceNetSmallV2Model.paddingIndex;
import static ramo.klevis.ml.recogntion.face.FaceNetSmallV2Model.reluIndex;

@Slf4j
public class FaceNetSmallV2Helper {
    static final String BASE = "FaceRecognition/src/main/resources/face/";

    static ActivationLayer relu() {
        return new ActivationLayer.Builder().activation(Activation.RELU).build();
    }

    static ZeroPaddingLayer zeroPadding(int i) {
        return new ZeroPaddingLayer.Builder(new int[]{i, i, i, i})
                .build();
    }

    static ConvolutionLayer convolution(int filterSize, int in, int out) {
        return new ConvolutionLayer.Builder(new int[]{filterSize, filterSize})
                .convolutionMode(ConvolutionMode.Truncate)
                .nIn(in).nOut(out)
                .build();
    }

    static ConvolutionLayer convolution(int filterSize, int in, int out, int strides) {
        return new ConvolutionLayer.Builder(new int[]{filterSize, filterSize}, new int[]{strides, strides})
                .convolutionMode(ConvolutionMode.Truncate)
                .nIn(in).nOut(out)
                .build();
    }

    static void convolution2dAndBN(ComputationGraphConfiguration.GraphBuilder graph, String layerName,
                                   Integer conv1Out, Integer conv1In, int[] conv1Filter, int[] conv1Strides,
                                   Integer conv2Out, Integer conv2in, int[] conv2Filter, int[] conv2Strides,
                                   int[] padding, String lastLayer) {

        String num = (conv2Out == null) ? "" : "1";

        graph.addLayer(layerName + "_conv" + num,
                new ConvolutionLayer.Builder(conv1Filter, conv1Strides).nIn(conv1In).nOut(conv1Out)
                        .convolutionMode(ConvolutionMode.Truncate).build(), lastLayer)

                .addLayer(layerName + "_bn" + num,
                        batchNorm(conv1Out),
                        layerName + "_conv" + num)

                .addLayer(nextReluId(),
                        relu(),
                        layerName + "_bn" + num);

        if (padding == null) {
            return;
        }
        graph.addLayer(nextPaddingId(),
                new ZeroPaddingLayer.Builder(padding)
                        .build(), lastReluId());
        if (conv2Out == null) {
            return;
        }
        graph.addLayer(layerName + "_conv2",
                new ConvolutionLayer.Builder(conv2Filter, conv2Strides).nIn(conv2in).nOut(conv2Out)
                        .convolutionMode(ConvolutionMode.Truncate).build(),
                lastPaddingId())

                .addLayer(layerName + "_bn2",
                        batchNorm(conv2Out),
                        layerName + "_conv2")

                .addLayer(nextReluId(),
                        relu(),
                        layerName + "_bn2");

    }

    static BatchNormalization batchNorm(int in) {
        return new BatchNormalization.Builder(false).eps(0.00001).nIn(in).nOut(in).build();
    }

    static String nextReluId() {
        return "relu" + reluIndex++;
    }

    static String nextPaddingId() {
        return "padding" + (paddingIndex++);
    }

    static String lastPaddingId() {
        return "padding" + (paddingIndex - 1);
    }

    static String lastReluId() {
        return "relu" + (reluIndex - 1);
    }

    static double[] readWightsValues(String path) throws IOException {
        String collect = Files.lines(Paths.get(path))
                .collect(Collectors.joining(","));
        return Arrays.stream(collect.split(",")).mapToDouble(Double::parseDouble).toArray();
    }

    static void loadWeights(ComputationGraph computationGraph) throws IOException {

        Layer[] layers = computationGraph.getLayers();
        for (Layer layer : layers) {
            List<double[]> all = new ArrayList<>();
            String layerName = layer.conf().getLayer().getLayerName();
            if (layerName.contains("bn")) {
                all.add(readWightsValues(BASE + layerName + "_w.csv"));
                all.add(readWightsValues(BASE + layerName + "_b.csv"));
                all.add(readWightsValues(BASE + layerName + "_m.csv"));
                all.add(readWightsValues(BASE + layerName + "_v.csv"));
                layer.setParams(mergeAll(all));
            } else if (layerName.contains("conv")) {
                all.add(readWightsValues(BASE + layerName + "_b.csv"));
                all.add(readWightsValues(BASE + layerName + "_w.csv"));
                layer.setParams(mergeAll(all));
            } else if (layerName.contains("dense")) {
                double[] w = readWightsValues(BASE + layerName + "_w.csv");
                all.add(w);
                double[] b = readWightsValues(BASE + layerName + "_b.csv");
                all.add(b);
                layer.setParams(mergeAll(all));
            }
        }
    }

    private static INDArray mergeAll(List<double[]> all) {
        INDArray[] allArr = new INDArray[all.size()];
        int index = 0;
        for (double[] doubles : all) {
            allArr[index++] = Nd4j.create(doubles);
        }
        return Nd4j.toFlattened(allArr);
    }

}