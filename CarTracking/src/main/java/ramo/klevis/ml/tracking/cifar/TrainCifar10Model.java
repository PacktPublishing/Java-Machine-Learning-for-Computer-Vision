package ramo.klevis.ml.tracking.cifar;

import org.bytedeco.javacpp.opencv_core;
import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.graph.L2NormalizeVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.CenterLossOutputLayer;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.DropoutLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.layers.objdetect.DetectedObject;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.nn.weights.WeightInit;
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
import org.nd4j.linalg.dataset.api.preprocessor.VGG16ImagePreProcessor;
import org.nd4j.linalg.learning.config.IUpdater;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import ramo.klevis.ml.tracking.FaceNetSmallV2Model;
import ramo.klevis.ml.tracking.ImageUtils;
import ramo.klevis.ml.tracking.yolo.Speed;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.Map;

public class TrainCifar10Model {

    public static final int NUM_POSSIBLE_LABELS = 611;
    public static final int BATCH_SIZE = 512;
    public static final int E_BATCH_SIZE = 512;
    private static final DataNormalization IMAGE_PRE_PROCESSOR = new CifarImagePreProcessor();
    private static final NativeImageLoader LOADER = new NativeImageLoader(ImageUtils.HEIGHT, ImageUtils.WIDTH, 3);
    private static final String CONTENT_LAYER_NAME = "embeddings";
    private static final String MODEL_SAVE_PATH = "CarTracking/src/main/resources/models/";
    private static final int SAVE_INTERVAL = 40;
    private static final int TEST_INTERVAL = 5;
    private static final int EPOCH_TRAINING = 2400;
    public static final int EMBEDDINGS = 512;
    public static final int I_EPOCH = 1120;
    private static final String PREFIX = "";
    private ComputationGraph cifar10Transfer;

    public static void main(String[] args) throws IOException {
        TrainCifar10Model trainCifar10Model = new TrainCifar10Model();
        trainCifar10Model.train();
    }

    public ComputationGraph getCifar10Transfer() {
        return cifar10Transfer;
    }

    private void train() throws IOException {

//        ZooModel zooModel = VGG16.builder().build();
//        ComputationGraph vgg16 = (ComputationGraph) zooModel.initPretrained(PretrainedType.CIFAR10);
//        System.out.println(vgg16.summary());

//        ZooModel zooModel = VGG16.builder().build();
//        ComputationGraph vgg16 = (ComputationGraph) zooModel.initPretrained(PretrainedType.IMAGENET);
//        System.out.println(vgg16.summary());

//        FaceNetSmallV2Model faceNetSmallV2Model = new FaceNetSmallV2Model();
//        ComputationGraph vgg16 = faceNetSmallV2Model.init();

        ComputationGraph vgg16 = ModelSerializer.restoreComputationGraph(
                new File(MODEL_SAVE_PATH + PREFIX + NUM_POSSIBLE_LABELS + "_epoch_data_e" + EMBEDDINGS + "_b" + E_BATCH_SIZE + "_" + I_EPOCH + ".zip"));
        IUpdater iUpdaterWithDefaultConfig = Updater.ADADELTA.getIUpdaterWithDefaultConfig();
//        iUpdaterWithDefaultConfig.setLrAndSchedule(0.1, null);
        FineTuneConfiguration fineTuneConf = new FineTuneConfiguration.Builder()
                .seed(1234)
//                .weightInit(WeightInit.XAVIER)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .activation(Activation.RELU)
                .updater(iUpdaterWithDefaultConfig)
                .cudnnAlgoMode(ConvolutionLayer.AlgoMode.NO_WORKSPACE)
                .miniBatch(true)
                .inferenceWorkspaceMode(WorkspaceMode.ENABLED)
                .trainingWorkspaceMode(WorkspaceMode.ENABLED)
                .pretrain(true)
                .backprop(true)
                .build();

        ComputationGraph cifar10 = new TransferLearning.GraphBuilder(vgg16)
                .setWorkspaceMode(WorkspaceMode.ENABLED)
                .fineTuneConfiguration(fineTuneConf)
                .setInputTypes(InputType.convolutionalFlat(ImageUtils.HEIGHT,
                        ImageUtils.WIDTH, 3))
                /* .removeVertexKeepConnections("fc2")
                 .removeVertexKeepConnections("predictions")
                 .addLayer("fc2", new DenseLayer.Builder()
                         .nIn(4096)
                         .nOut(EMBEDDINGS)
                         .activation(Activation.RELU).build(), "fc1")
                 .setFeatureExtractor("block3_pool")*/


//                .removeVertexAndConnections("dense_2_loss")
//                .removeVertexAndConnections("dense_2")
//                .removeVertexAndConnections("dense_1")
//                .removeVertexAndConnections("dropout_1")
//                .removeVertexAndConnections("embeddings")
//                .removeVertexAndConnections("lossLayer")
////                .removeVertexAndConnections("flatten_1")
//                .addLayer("dense_1", new DenseLayer.Builder()
//                        .nIn(4096)
//                        .nOut(EMBEDDINGS)
//                        .activation(Activation.RELU).build(), "flatten_1")
//                .addVertex("embeddings", new L2NormalizeVertex(new int[]{}, 1e-12), "fc2")
//                .addLayer("lossLayer", new CenterLossOutputLayer.Builder()
//                                .lossFunction(LossFunctions.LossFunction.SQUARED_LOSS)
//                                .activation(Activation.SOFTMAX).nIn(EMBEDDINGS).nOut(NUM_POSSIBLE_LABELS).lambda(1e-4).alpha(0.9)
//                                .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer).build(),
//                        "embeddings")
//                .setOutputs("lossLayer")
                .build();

        System.out.println(cifar10.summary());
        File rootDir = new File("CarTracking/train_from_video_" + NUM_POSSIBLE_LABELS);
        DataSetIterator dataSetIterator = ImageUtils.createDataSetIterator(rootDir, NUM_POSSIBLE_LABELS, BATCH_SIZE);
        DataSetIterator testIterator = ImageUtils.createDataSetIterator(rootDir, NUM_POSSIBLE_LABELS, BATCH_SIZE);
        cifar10.setListeners(new ScoreIterationListener(2));
        int iEpoch = I_EPOCH;
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
            }
            iEpoch++;

            String modelName = PREFIX + NUM_POSSIBLE_LABELS + "_epoch_data_e" + EMBEDDINGS + "_b" + BATCH_SIZE + "_" + iEpoch + ".zip";
            if (iEpoch % SAVE_INTERVAL == 0) {
                ModelSerializer.writeModel(cifar10,
                        new File(MODEL_SAVE_PATH +
                                modelName),
                        false);
            }
            if (iEpoch % TEST_INTERVAL == 0) {
                Evaluation eval = cifar10.evaluate(testIterator);
                System.out.println(eval.stats());
                testIterator.reset();
            }
//            TestModels.TestResult test = TestModels.test(cifar10, modelName);
//            System.out.println("Test Results >> " + test);
            dataSetIterator.reset();
            System.out.println("iEpoch = " + iEpoch);
        }
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
