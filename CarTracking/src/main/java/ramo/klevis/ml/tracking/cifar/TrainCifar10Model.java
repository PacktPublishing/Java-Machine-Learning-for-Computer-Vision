package ramo.klevis.ml.tracking.cifar;

import org.bytedeco.javacpp.opencv_core;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.graph.L2NormalizeVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.CenterLossOutputLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.layers.objdetect.DetectedObject;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.deeplearning4j.zoo.PretrainedType;
import org.deeplearning4j.zoo.ZooModel;
import org.deeplearning4j.zoo.model.VGG16;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.learning.config.IUpdater;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import ramo.klevis.ml.tracking.ImageUtils;
import ramo.klevis.ml.tracking.yolo.Speed;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.Map;

import static org.datavec.image.loader.CifarLoader.CHANNELS;

public class TrainCifar10Model {

    private static final int HEIGHT = 32;
    private static final int WIDTH = 32;
    private static final DataNormalization IMAGE_PRE_PROCESSOR = new CifarImagePreProcessor();
    private static final ParentPathLabelGenerator LABEL_GENERATOR_MAKER = new ParentPathLabelGenerator();
    private static final NativeImageLoader LOADER = new NativeImageLoader(HEIGHT, WIDTH, 3);

    private static final String CONTENT_LAYER_NAME = "embeddings";
    private static final String MODEL_SAVE_PATH = "CarTracking/src/main/resources/models/";

    private static final int NUM_POSSIBLE_LABELS = 611;
    private static final int BATCH_SIZE = 64;
    private static final int SAVE_INTERVAL = 5;
    private static final int TEST_INTERVAL = 1;
    private static final int EPOCH_TRAINING = 100;
    private ComputationGraph cifar10Transfer;

    public static void main(String[] args) throws IOException {
        TrainCifar10Model trainCifar10Model = new TrainCifar10Model();
        trainCifar10Model.train();
    }

    public ComputationGraph getCifar10Transfer() {
        return cifar10Transfer;
    }

    private void train() throws IOException {

        ZooModel zooModel = VGG16.builder().build();
        ComputationGraph vgg16 = (ComputationGraph) zooModel.initPretrained(PretrainedType.CIFAR10);
        System.out.println(vgg16.summary());
        IUpdater iUpdaterWithDefaultConfig = Updater.ADAM.getIUpdaterWithDefaultConfig();
        FineTuneConfiguration fineTuneConf = new FineTuneConfiguration.Builder()
                .seed(1234)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .activation(Activation.RELU)
                .updater(iUpdaterWithDefaultConfig)
                .miniBatch(true)
                .inferenceWorkspaceMode(WorkspaceMode.ENABLED)
                .trainingWorkspaceMode(WorkspaceMode.ENABLED)
                .backprop(true)
                .build();

        ComputationGraph cifar10 = new TransferLearning.GraphBuilder(vgg16)
                .setWorkspaceMode(WorkspaceMode.ENABLED)
                .fineTuneConfiguration(fineTuneConf)
                .setInputTypes(InputType.convolutionalFlat(HEIGHT, WIDTH, 3))
                .removeVertexAndConnections("dense_2_loss")
                .removeVertexKeepConnections("dense_2")
                .removeVertexKeepConnections("dense_1")
                .removeVertexKeepConnections("dropout_1")
                .addLayer("dense_1", new DenseLayer.Builder()
                        .nIn(4096)
                        .nOut(1024)
                        .activation(Activation.RELU).build(), "flatten_1")
                .addVertex("embeddings", new L2NormalizeVertex(new int[]{}, 1e-12), "dense_1")
                .removeVertexKeepConnections("lossLayer")
                .addLayer("lossLayer", new CenterLossOutputLayer.Builder()
                                .lossFunction(LossFunctions.LossFunction.SQUARED_LOSS)
                                .activation(Activation.SOFTMAX).nIn(1024).nOut(NUM_POSSIBLE_LABELS).lambda(1e-4).alpha(0.9)
                                .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer).build(),
                        "embeddings")
                .setOutputs("lossLayer")
                .build();
        System.out.println(cifar10.summary());
        File rootDir = new File("train_from_videos");
        System.out.println(rootDir.getAbsolutePath());
        DataSetIterator dataSetIterator = createDataSetIterator(rootDir);
        DataSetIterator testIterator = createDataSetIterator(rootDir);
        cifar10.setListeners(new ScoreIterationListener(1));
        int iEpoch = 0;
        int iIteration = 0;
        while (iEpoch < EPOCH_TRAINING) {
            while (dataSetIterator.hasNext()) {
                DataSet trainMiniBatchData = null;
                try {
                    trainMiniBatchData = dataSetIterator.next();
                } catch (Exception e) {
                    e.printStackTrace();
                }
                cifar10.fit(trainMiniBatchData);
                iIteration++;
                System.out.println("iIteration = " + iIteration);
            }
            iEpoch++;

            if (iEpoch % SAVE_INTERVAL == 0) {
                ModelSerializer.writeModel(cifar10,
                        new File(MODEL_SAVE_PATH +
                                NUM_POSSIBLE_LABELS + "_epoch_data_n1024d_b256_" + iEpoch + ".zip"),
                        false);
            }
            if (iEpoch % TEST_INTERVAL == 0) {
                Evaluation eval = cifar10.evaluate(testIterator);
                System.out.println(eval.stats());
                testIterator.reset();
            }
            TestModels.TestResult test = TestModels.test(cifar10);
            System.out.println("Test Results >> " + test);
            dataSetIterator.reset();
            System.out.println("iEpoch = " + iEpoch);
        }
    }

    private DataSetIterator createDataSetIterator(File sample) throws IOException {
        ImageRecordReader imageRecordReader = new ImageRecordReader(HEIGHT, WIDTH, CHANNELS, LABEL_GENERATOR_MAKER);
        imageRecordReader.initialize(new FileSplit(sample));
        DataSetIterator iterator = new RecordReaderDataSetIterator(imageRecordReader, BATCH_SIZE,
                1, NUM_POSSIBLE_LABELS);
        iterator.setPreProcessor(new CifarImagePreProcessor());
        return iterator;
    }


    public void loadTrainedModel() throws IOException {
        cifar10Transfer = ModelSerializer.
                restoreComputationGraph(new File(MODEL_SAVE_PATH +
                        "611_epoch_data_n1024d_b64_545.zip"));
        System.out.println(cifar10Transfer.summary());
    }

    public INDArray getEmbeddings(opencv_core.Mat file, DetectedObject obj, Speed selectedSpeed) throws Exception {
        BufferedImage croppedImage = ImageUtils.cropImageWithYolo(selectedSpeed,
                file, obj, false);
        INDArray croppedContent = LOADER.asMatrix(croppedImage);
        IMAGE_PRE_PROCESSOR.transform(croppedContent);

        Map<String, INDArray> stringINDArrayMap = getCifar10Transfer()
                .feedForward(croppedContent, false);
        INDArray indArray = stringINDArrayMap.get(CONTENT_LAYER_NAME);

        return indArray;
    }
}
