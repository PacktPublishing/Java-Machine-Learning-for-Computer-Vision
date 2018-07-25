package ramo.klevis.ml.vg16;

import lombok.extern.slf4j.Slf4j;
import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.VGG16ImagePreProcessor;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;

/**
 * Created by Klevis Ramo
 */
@Slf4j
public class CatVsDogRecognition {
    public static final String TRAINED_PATH_MODEL = DataStorage.DATA_PATH + "/model.zip";
    private ComputationGraph computationGraph;

    public CatVsDogRecognition() throws IOException {
        this.computationGraph = loadModel();
        computationGraph.init();
        log.info(computationGraph.summary());
    }

    public AnimalType detectAnimalType(File file, Double threshold) throws IOException {
        INDArray image = imageFileToMatrix(file);
        INDArray output = computationGraph.outputSingle(false, image);
        if (output.getDouble(0) > threshold) {
            return AnimalType.CAT;
        } else if (output.getDouble(1) > threshold) {
            return AnimalType.DOG;
        } else {
            return AnimalType.NOT_KNOWN;
        }
    }

    private INDArray imageFileToMatrix(File file) throws IOException {
        NativeImageLoader loader = new NativeImageLoader(224, 224, 3);
        INDArray image = loader.asMatrix(new FileInputStream(file));
        DataNormalization dataNormalization = new VGG16ImagePreProcessor();
        dataNormalization.transform(image);
        return image;
    }

    public ComputationGraph loadModel() throws IOException {
        computationGraph = ModelSerializer.restoreComputationGraph(new File(TRAINED_PATH_MODEL));
        return computationGraph;
    }

}
