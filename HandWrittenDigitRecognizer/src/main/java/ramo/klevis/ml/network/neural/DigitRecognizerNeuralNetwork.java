package ramo.klevis.ml.network.neural;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import ramo.klevis.ml.LabeledImage;

import java.io.File;
import java.io.IOException;

/**
 * Created by Klevis Ramo
 */
@Slf4j
public class DigitRecognizerNeuralNetwork {

    private static final String OUTPUT_DIRECTORY = "HandWrittenDigitRecognizer/src/main/resources/trainedModel.zip";

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
     * Number of total traverses through data.
     * with 5 epochs we will have 5/@MINI_BATCH_SIZE iterations or weights updates
     */
    private static final int EPOCHS = 5;

    /**
     * The alpha learning rate defining the size of step towards the minimum
     */
    private static final double LEARNING_RATE = 0.01;

    /**
     * https://en.wikipedia.org/wiki/Random_seed
     */
    private static final int SEED = 123;
    private static final int IMAGE_WIDTH = 28;
    private static final int IMAGE_HEIGHT = 28;


    public void train() throws Exception {

        /*
            Create an iterator using the batch size for one iteration
         */
        log.info("Load data....");
        DataSetIterator mnistTrain = new MnistDataSetIterator(MINI_BATCH_SIZE, true, SEED);

        /*
            Construct the neural neural
         */
        log.info("Build model....");


        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(SEED)
                .learningRate(LEARNING_RATE)
                .weightInit(WeightInit.XAVIER)
                //NESTEROVS is referring to gradient descent with momentum
                .updater(Updater.NESTEROVS)
                .list()
                /**
                 * First hidden layer uses 128 neurons each with RELU activation.
                 * Is called dense layer because every neuron is linked with
                 * every other neuron on next layer and previous layer
                 */
                .layer(0, new DenseLayer.Builder().activation(Activation.RELU)
                        .nIn(IMAGE_WIDTH * IMAGE_HEIGHT)
                        .nOut(128).build())
                /**
                 * Second hidden layer uses 64 neurons each with RELU activation.
                 */
                .layer(1, new DenseLayer.Builder().activation(Activation.RELU)
                        .nOut(64).build())
                /**
                 * The output layer using SOFTMAX to predict 10 classes(OUTPUT)
                 * NEGATIVELOGLIKELIHOOD is just a cost function measuring how
                 * good our prediction or hypothesis is doing against real digits
                 */
                .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(OUTPUT)
                        .activation(Activation.SOFTMAX)
                        .build())
                .backprop(true)
                .pretrain(false)
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();


        log.info("Train model....");
        model.setListeners(new ScoreIterationListener(100));
        DataSetIterator mnistTest = null;
        for (int i = 0; i < EPOCHS; i++) {
            model.fit(mnistTrain);
            log.info("*** Completed epoch ***", i);
            if (mnistTest == null) {
                mnistTest = new MnistDataSetIterator(MINI_BATCH_SIZE, false, SEED);
            }
            log.info("Evaluate model....");
            Evaluation eval = model.evaluate(mnistTest);
            log.info(eval.stats());
            if (eval.accuracy() >= 0.97) {
                /**
                 * //Where to save the neural.
                 * Note: the file is in .zip format - can be opened externally
                 */
                File locationToSave = new File(OUTPUT_DIRECTORY);
                log.info("Saving model at " + locationToSave.getAbsolutePath());
                ModelSerializer.writeModel(model, locationToSave, true);
                log.info("Congratulations,the desired score found,!");
                break;
            }
            mnistTest.reset();
        }


        log.info("****************Example finished********************");
    }

    private MultiLayerNetwork preTrainedModel;

    public void init() throws IOException {
        preTrainedModel = ModelSerializer.restoreMultiLayerNetwork(new File(OUTPUT_DIRECTORY));
    }

    public int predict(LabeledImage labeledImage) {
        int[] predict = preTrainedModel.predict(Nd4j.create(labeledImage.getNormalizedPixels()));
        return predict[0];
    }

    public static void main(String[] args) throws Exception {
        new DigitRecognizerNeuralNetwork().train();
    }

}