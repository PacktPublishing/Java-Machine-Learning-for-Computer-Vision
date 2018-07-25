package ramo.klevis.ml.vg16;

import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.BaseImageLoader;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.VGG16ImagePreProcessor;

import java.io.File;
import java.io.IOException;
import java.util.Random;

/**
 * Created by Klevis Ramo
 */
public interface DataStorage {

    Random RAND_NUM_GEN = new Random(12345);
    String[] ALLOWED_FORMATS = BaseImageLoader.ALLOWED_FORMATS;
    ParentPathLabelGenerator LABEL_GENERATOR_MAKER = new ParentPathLabelGenerator();
    BalancedPathFilter PATH_FILTER = new BalancedPathFilter(RAND_NUM_GEN, ALLOWED_FORMATS, LABEL_GENERATOR_MAKER);

    int HEIGHT = 224;
    int WIDTH = 224;
    int CHANNELS = 3;

    String DATA_PATH = "CatVsDogRecognition/src/main/resources";
    String SAVING_PATH = DATA_PATH + "/saved/modelIteration_";
    String TEST_DIRECTORY_PATH = DATA_PATH + "/test_both";
    String TRAIN_DIRECTORY_PATH = DATA_PATH + "/train_both";

    File TRAIN_DIRECTORY_DATA = new File(TRAIN_DIRECTORY_PATH);
    FileSplit TRAIN_DATA = new FileSplit(TRAIN_DIRECTORY_DATA, NativeImageLoader.ALLOWED_FORMATS, RAND_NUM_GEN);

    File TEST_DIRECTORY_DATA = new File(TEST_DIRECTORY_PATH);
    FileSplit TEST_DATA = new FileSplit(TEST_DIRECTORY_DATA, NativeImageLoader.ALLOWED_FORMATS, RAND_NUM_GEN);

    String ONLINE_DATA_URL = "https://dl.dropboxusercontent.com/s/tqnp49apphpzb40/dataTraining.zip?dl=0";

    /**
     * Number prediction classes.
     * To predict if an image is a cat or dog we need to classes
     */
    int NUM_POSSIBLE_LABELS = 2;

    /**
     * Mini batch gradient descent size or number of matrices processed in parallel.
     * For CORE-I7 16 is good for GPU please change to 128 and up
     */
    int BATCH_SIZE = 16;
    /**
     * The parentage of available data used for training.
     * 85% for training and 15% for development(dev) data set
     * Note: There is also a test data set where we test after each epoch
     * how the network generalizes to unbiased data
     */
    int TRAIN_SIZE = 85;


    default NeuralNetworkTrainingData loadData() throws IOException {
        InputSplit[] trainAndDevData = TRAIN_DATA.sample(PATH_FILTER, TRAIN_SIZE, 100 - TRAIN_SIZE);
        return new NeuralNetworkTrainingData(getDataSetIterator(trainAndDevData[0]),
                getDataSetIterator(trainAndDevData[1]),
                getDataSetIterator(TEST_DATA.sample(PATH_FILTER, 1, 0)[0]));

    }

    default DataSetIterator getDataSetIterator(InputSplit sample) throws IOException {
        ImageRecordReader imageRecordReader = new ImageRecordReader(HEIGHT, WIDTH, CHANNELS, LABEL_GENERATOR_MAKER);
        imageRecordReader.initialize(sample);

        DataSetIterator iterator = new RecordReaderDataSetIterator(imageRecordReader, BATCH_SIZE, 1, NUM_POSSIBLE_LABELS);
        iterator.setPreProcessor(new VGG16ImagePreProcessor());
        return iterator;
    }
}
