package ramo.klevis.ml.vg16;

import lombok.extern.slf4j.Slf4j;
import org.apache.ant.compress.taskdefs.Unzip;
import org.apache.commons.io.FileUtils;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.deeplearning4j.zoo.PretrainedType;
import org.deeplearning4j.zoo.ZooModel;
import org.deeplearning4j.zoo.model.VGG16;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.IOException;
import java.net.URL;
import java.util.zip.Adler32;

import static ramo.klevis.ml.vg16.DataStorage.DATA_PATH;
import static ramo.klevis.ml.vg16.DataStorage.NUM_POSSIBLE_LABELS;
import static ramo.klevis.ml.vg16.DataStorage.ONLINE_DATA_URL;
import static ramo.klevis.ml.vg16.DataStorage.SAVING_PATH;
import static ramo.klevis.ml.vg16.DataStorage.TRAIN_DIRECTORY_PATH;

/**
 * Created by Klevis Ramo.
 */
@Slf4j
public class TransferLearningVGG16 {


    private static final int SAVING_INTERVAL = 100;

    /**
     * Number of total traverses through data.
     * with 5 epochs we will have 5/@MINI_BATCH_SIZE iterations or weights updates
     */
    private static final int EPOCH = 5;


    /**
     * The layer where we need to stop back propagating
     */
    private static final String FREEZE_UNTIL_LAYER = "fc2";

    /**
     * The alpha learning rate defining the size of step towards the minimum
     */
    private static final double LEARNING_RATE = 5e-5;

    private NeuralNetworkTrainingData neuralNetworkTrainingData;


    public static void main(String[] args) throws IOException {
        TransferLearningVGG16 transferLearningVGG16 = new TransferLearningVGG16();
        transferLearningVGG16.train();
    }

    public void train() throws IOException {
        ComputationGraph preTrainedNet = loadVGG16PreTrainedWeights();
        log.info("VGG 16 Architecture");
        log.info(preTrainedNet.summary());

        log.info("Start Downloading NeuralNetworkTrainingData...");

        downloadAndUnzipDataForTheFirstTime();

        log.info("NeuralNetworkTrainingData Downloaded and unzipped");

        neuralNetworkTrainingData = new DataStorage() {
        }.loadData();

        FineTuneConfiguration fineTuneConf = new FineTuneConfiguration.Builder()
                .learningRate(LEARNING_RATE)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(Updater.NESTEROVS)
                .seed(1234)
                .build();

        ComputationGraph vgg16Transfer = new TransferLearning.GraphBuilder(preTrainedNet)
                .fineTuneConfiguration(fineTuneConf)
                .setFeatureExtractor(FREEZE_UNTIL_LAYER)
                .removeVertexKeepConnections("predictions")
                .addLayer("predictions",
                        new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                                .nIn(4096)
                                .nOut(NUM_POSSIBLE_LABELS)
                                .weightInit(WeightInit.XAVIER)
                                .activation(Activation.SOFTMAX)
                                .build(),
                        FREEZE_UNTIL_LAYER)
                .build();
        vgg16Transfer.setListeners(new ScoreIterationListener(5));

        log.info("Modified VGG 16 Architecture for transfer learning");
        log.info(vgg16Transfer.summary());

        int iEpoch = 0;
        int iIteration = 0;
        while (iEpoch < EPOCH) {
            while (neuralNetworkTrainingData.getTrainIterator().hasNext()) {
                DataSet trainMiniBatchData = neuralNetworkTrainingData.getTrainIterator().next();
                vgg16Transfer.fit(trainMiniBatchData);
                saveProgressEveryConfiguredInterval(vgg16Transfer, iEpoch, iIteration);
                iIteration++;
            }
            neuralNetworkTrainingData.getTrainIterator().reset();
            iEpoch++;

            evalOn(vgg16Transfer, neuralNetworkTrainingData.getTestIterator(), iEpoch);
        }
    }


    private void saveProgressEveryConfiguredInterval(ComputationGraph vgg16Transfer, int iEpoch, int
            iIteration) throws IOException {
        if (iIteration % SAVING_INTERVAL == 0 && iIteration != 0) {

            ModelSerializer.writeModel(vgg16Transfer, new File(SAVING_PATH + iIteration + "_epoch_" + iEpoch + ".zip"),
                    false);
            evalOn(vgg16Transfer, neuralNetworkTrainingData.getDevIterator(), iIteration);
        }
    }

    private ComputationGraph loadVGG16PreTrainedWeights() throws IOException {
        ZooModel zooModel = new VGG16();
        log.info("Start Downloading VGG16 model...");
        return (ComputationGraph) zooModel.initPretrained(PretrainedType.IMAGENET);
    }

    private void unzip(File fileZip) {

        Unzip unZip = new Unzip();
        unZip.setSrc(fileZip);
        unZip.setDest(new File(DATA_PATH));
        unZip.execute();
    }

    private void downloadAndUnzipDataForTheFirstTime() throws IOException {
        File data = new File(DATA_PATH + "/data.zip");
        if (!data.exists() || FileUtils.checksum(data, new Adler32()).getValue() != 1195241806) {
            data.delete();
            FileUtils.copyURLToFile(new URL(ONLINE_DATA_URL), data);
            log.info("File downloaded");
        }
        if (!new File(TRAIN_DIRECTORY_PATH).exists()) {
            log.info("Unzipping NeuralNetworkTrainingData...");
            unzip(data);
        }
    }


    private void evalOn(ComputationGraph vgg16Transfer, DataSetIterator testIterator, int iEpoch) throws IOException {
        log.info("Evaluate model at iteration " + iEpoch + " ....");
        Evaluation eval = vgg16Transfer.evaluate(testIterator);
        log.info(eval.stats());
        testIterator.reset();

    }

}
