package ramo.klevis.ml.network.neural;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration;
import org.deeplearning4j.earlystopping.EarlyStoppingResult;
import org.deeplearning4j.earlystopping.saver.LocalFileModelSaver;
import org.deeplearning4j.earlystopping.termination.MaxEpochsTerminationCondition;
import org.deeplearning4j.earlystopping.trainer.EarlyStoppingTrainer;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import ramo.klevis.ml.LabeledImage;

import java.io.File;
import java.io.IOException;

/**
 * Created by Klevis Ramo.
 */
@Slf4j
public class DigitRecognizerConvolutionalNeuralNetwork {

    private static final String OUT_DIR = "HandWrittenDigitRecognizer/src/main/resources/cnnCurrentTrainingModels";
    private static final String TRAINED_MODEL_FILE = "HandWrittenDigitRecognizer/src/main/resources/cnnTrainedModels/bestModel.bin";
    private MultiLayerNetwork preTrainedModel;

    private static final int CHANNELS = 1;
    /**
     * Number prediction classes.
     * We have 0-9 digits so 10 classes in total.
     */
    private static final int OUTPUT = 10;
    /**
     * Mini batch gradient descent size or number of matrices processed in parallel.
     * For CORE-I7 16 is good for GPU please change to 128 and up
     */
    private static final int MINI_BATCH_SIZE = 16;// Number of training epochs
    /**
     * Number of total traverses through data. In this case it is used as the maximum epochs we allow
     * with 5 epochs we will have 5/@MINI_BATCH_SIZE iterations or weights updates
     */
    private static final int MAX_EPOCHS = 20;

    /**
     * The alpha learning rate defining the size of step towards the minimum
     */

    private static final double LEARNING_RATE = 0.01;

    /**
     * https://en.wikipedia.org/wiki/Random_seed
     */
    private static final int SEED = 123;

    public DigitRecognizerConvolutionalNeuralNetwork() throws IOException {
        init();
    }

    public void init() throws IOException {
        preTrainedModel = ModelSerializer.restoreMultiLayerNetwork(new File(TRAINED_MODEL_FILE));
    }

    public int predict(LabeledImage labeledImage) {
        int[] predict = preTrainedModel.predict(Nd4j.create(labeledImage.getNormalizedPixels()));
        return predict[0];
    }

    public void train() throws IOException {

        MnistDataSetIterator mnistTrain = new MnistDataSetIterator(MINI_BATCH_SIZE, true, 12345);

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(SEED)
                .learningRate(LEARNING_RATE)
                .weightInit(WeightInit.XAVIER)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(Updater.NESTEROVS)
                .list()
                .layer(0, new ConvolutionLayer.Builder(5, 5)
                        .nIn(CHANNELS)
                        .stride(1, 1)
                        .nOut(20)
                        .activation(Activation.IDENTITY)
                        .build())
                .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .build())
                .layer(2, new ConvolutionLayer.Builder(5, 5)
                        .nIn(20)
                        .stride(1, 1)
                        .nOut(50)
                        .activation(Activation.IDENTITY)
                        .build())
                .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .build())
                .layer(4, new DenseLayer.Builder().activation(Activation.RELU)
                        .nIn(800)
                        .nOut(128).build())
                .layer(5, new DenseLayer.Builder().activation(Activation.RELU)
                        .nIn(128)
                        .nOut(64).build())
                .layer(6, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(OUTPUT)
                        .activation(Activation.SOFTMAX)
                        .build())
                .setInputType(InputType.convolutionalFlat(28, 28, 1))
                .backprop(true).pretrain(false).build();

        EarlyStoppingConfiguration earlyStoppingConfiguration = new EarlyStoppingConfiguration.Builder()
                .epochTerminationConditions(new MaxEpochsTerminationCondition(MAX_EPOCHS))
                .scoreCalculator(new AccuracyCalculator(new MnistDataSetIterator(MINI_BATCH_SIZE, false, 12345)))
                .evaluateEveryNEpochs(1)
                .modelSaver(new LocalFileModelSaver(OUT_DIR))
                .build();

        EarlyStoppingTrainer trainer = new EarlyStoppingTrainer(earlyStoppingConfiguration, conf, mnistTrain);

        EarlyStoppingResult<MultiLayerNetwork> result = trainer.fit();

        log.info("Termination reason: " + result.getTerminationReason());
        log.info("Termination details: " + result.getTerminationDetails());
        log.info("Total epochs: " + result.getTotalEpochs());
        log.info("Best epoch number: " + result.getBestModelEpoch());
        log.info("Score at best epoch: " + result.getBestModelScore());
    }

    public static void main(String[] args) throws Exception {
        new DigitRecognizerConvolutionalNeuralNetwork().train();
    }
}