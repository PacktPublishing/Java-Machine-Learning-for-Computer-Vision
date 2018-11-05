package ramo.klevis.ml.tracking.cifar;

import lombok.AllArgsConstructor;
import lombok.Data;
import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.util.ModelSerializer;
import org.jetbrains.annotations.NotNull;
import org.nd4j.linalg.api.ndarray.INDArray;
import ramo.klevis.ml.tracking.cifar.CifarImagePreProcessor;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;


/**
 * Created by klevis.ramo on 10/19/2018
 */

public class TestModels {

    private static final String BASE = "AutonomousDriving/src/main/resources/models/";
    private static final String TEST_DATA = "test_data";
    private static final double THRESHOLD = 0.85;
    private static final CifarImagePreProcessor IMAGE_PRE_PROCESSOR = new CifarImagePreProcessor();
    private static final int SIZE = 32;
    private static final NativeImageLoader LOADER = new NativeImageLoader(SIZE, SIZE, 3);

    public static void main(String[] args) throws IOException {
        String[] array = new String[]{
                "611_epoch_data_n1024d_b128_400.zip",
                "611_epoch_data_n1024d_b128_505.zip",
                "611_epoch_data_n1024d385.zip",
                "611_epoch_data_n1024d390.zip",
                "611_epoch_data_n1024d395.zip",
                "611_epoch_data_n1024d400.zip",
                "611_epoch_data_n1024d405.zip",
                "611_epoch_data_n1024d_b64_520.zip",
                "611_epoch_data_n1024d_b64_525.zip",
                "611_epoch_data_n1024d_b64_530.zip",
                "611_epoch_data_n1024d_b64_535.zip",
                "611_epoch_data_n1024d_b64_540.zip",
                "611_epoch_data_n1024d_b64_545.zip",
                "611_epoch_data_n1024d_b64_550.zip",
                "611_epoch_data_n1024d_b64_555.zip",
                "611_epoch_data_n1024d_b64_560.zip",
                "611_epoch_data_n1024d_b64_565.zip",
                "611_epoch_data_n1024d_b64_570.zip",
                "611_epoch_data_n1024d_b64_575.zip",
                "611_epoch_data_n1024d_b64_580.zip"};
        for (String s : array) {
            ComputationGraph vgg16 = ModelSerializer.restoreComputationGraph(
                    new File(BASE + s));
            TestResult testResult = test(vgg16);
            System.out.println(testResult);
        }
    }

    public static TestResult test(ComputationGraph vgg16) throws IOException {
        HashMap<File, List<INDArray>> map = buildEmbeddings(vgg16);

        Set<Map.Entry<File, List<INDArray>>> entries = map.entrySet();
        int wrongPredictionsWithOtherClasses = 0;
        int wrongPredictionsInsideOneClassOrdered = 0;
        int wrongPredictionsInsideOneClassNotOrdered = 0;
        for (Map.Entry<File, List<INDArray>> entry : entries) {
            wrongPredictionsWithOtherClasses += compareWithOtherClasses(entries, entry);
            wrongPredictionsInsideOneClassOrdered += compareInsideOneClassOrdered(entry);
            wrongPredictionsInsideOneClassNotOrdered += compareInsideOneClassNotOrdered(entry);
        }
        return new TestResult(wrongPredictionsInsideOneClassOrdered,
                wrongPredictionsInsideOneClassNotOrdered,
                wrongPredictionsWithOtherClasses,
                sum(wrongPredictionsWithOtherClasses,
                        wrongPredictionsInsideOneClassOrdered,
                        wrongPredictionsInsideOneClassNotOrdered));
    }

    private static int sum(int wrongPredictionsWithOtherClasses, int wrongPredictionsInsideOneClassOrdered, int wrongPredictionsInsideOneClassNotOrdered) {
        return wrongPredictionsInsideOneClassOrdered + wrongPredictionsInsideOneClassNotOrdered +
                wrongPredictionsWithOtherClasses;
    }

    private static int compareInsideOneClassOrdered(Map.Entry<File, List<INDArray>> entry) {
        int wrongPredictionsInsideOneClassOrdered = 0;
        List<INDArray> value = entry.getValue();
        INDArray prevEmbeddings = value.get(0);
        for (int i = 1; i < value.size(); i++) {
            if (prevEmbeddings.distance2(value.get(i)) >= THRESHOLD) {
                wrongPredictionsInsideOneClassOrdered++;
            }
            prevEmbeddings = value.get(i);
        }
        return wrongPredictionsInsideOneClassOrdered;
    }

    private static int compareInsideOneClassNotOrdered(Map.Entry<File, List<INDArray>> entry) {
        int wrongPredictionsInsideOneClassNotOrdered = 0;
        List<INDArray> value = entry.getValue();
        Collections.shuffle(value);
        for (INDArray indArray : value) {
            for (INDArray array : value) {
                if (indArray.distance2(array) >= THRESHOLD) {
                    wrongPredictionsInsideOneClassNotOrdered++;
                }
            }
        }
        return wrongPredictionsInsideOneClassNotOrdered;
    }


    private static int compareWithOtherClasses(Set<Map.Entry<File, List<INDArray>>> entries, Map.Entry<File, List<INDArray>> entry) {
        int misMatch = 0;
        File folder = entry.getKey();
        List<INDArray> currentEmbeddingsList = entry.getValue();
        for (INDArray currentEmbedding : currentEmbeddingsList) {
            for (Map.Entry<File, List<INDArray>> entryOther : entries) {
                if (entryOther.getKey().getName().equals(folder.getName())) {
                    continue;
                }
                List<INDArray> otherEmbeddingsList = entryOther.getValue();
                for (INDArray otherEmbedding : otherEmbeddingsList) {
                    if (currentEmbedding.distance2(otherEmbedding) < THRESHOLD) {
                        misMatch++;
                    }
                }
            }
        }
        return misMatch;
    }

    @NotNull
    private static HashMap<File, List<INDArray>> buildEmbeddings(ComputationGraph vgg16) throws IOException {
        File[] folders = new File(TEST_DATA).listFiles();
        HashMap<File, List<INDArray>> map = new HashMap<>();
        for (File folder : folders) {
            File[] carsInOneClass = folder.listFiles();
            Arrays.sort(carsInOneClass, (f1, f2) -> Long.valueOf(f1.lastModified())
                    .compareTo(f2.lastModified()));
            ArrayList<INDArray> indArrays = new ArrayList<>();
            for (File imageInsideOneClass : carsInOneClass) {
                indArrays.add(getEmbeddings(vgg16, imageInsideOneClass));
            }
            map.put(folder, indArrays);
        }
        return map;
    }

    private static INDArray getEmbeddings(ComputationGraph vgg16, File image) throws IOException {
        INDArray indArray = LOADER.asMatrix(image);
        IMAGE_PRE_PROCESSOR.preProcess(indArray);
        Map<String, INDArray> stringINDArrayMap = vgg16.feedForward(indArray, false);
        INDArray embeddings = stringINDArrayMap.get("embeddings");
        return embeddings;
    }

    @AllArgsConstructor
    @Data
    static class TestResult {
        int wrongPredictionsInsideOneClassOrdered;
        int wrongPredictionsInsideOneClassNotOrdered;
        int wrongPredictionsWithOtherClasses;
        int total;
    }

}
