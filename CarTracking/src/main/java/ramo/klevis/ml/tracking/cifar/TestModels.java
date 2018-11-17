package ramo.klevis.ml.tracking.cifar;

import lombok.AllArgsConstructor;
import lombok.Data;
import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.util.ModelSerializer;
import org.jetbrains.annotations.NotNull;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import ramo.klevis.ml.tracking.ImageUtils;

import java.io.File;
import java.io.IOException;
import java.util.*;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.stream.Stream;


/**
 * Created by klevis.ramo on 10/19/2018
 */

public class TestModels {

    private static final String BASE = "CarTracking/src/main/resources/models/";
    private static final String TEST_DATA = "CarTracking/test_data";
    private static final double THRESHOLD = 0.85;
    private static final CifarImagePreProcessor IMAGE_PRE_PROCESSOR = new CifarImagePreProcessor();
    private static final NativeImageLoader LOADER = new NativeImageLoader(ImageUtils.HEIGHT, ImageUtils.WIDTH, 3);
    private static boolean showTrainingPrecision = false;

    public static void main(String[] args) throws IOException {
        String[] allModels = new File(BASE).list();
//        allModels = new String[]{"96_data.zip",
//                "189_epoch_data_n189d_b64_135.zip",
//                "611_epoch_data_e1024_b512_180.zip"};
        for (String model : allModels) {
            ComputationGraph vgg16 = ModelSerializer.restoreComputationGraph(
                    new File(BASE + model));

            String classesNumber = model.substring(0, model.indexOf("_"));
            if (showTrainingPrecision) {
                showTrainingPrecision(vgg16, classesNumber);
            }
//            System.out.println(vgg16.summary());
            TestResult testResult = test(vgg16, model);
            System.out.println(testResult);
        }
    }

    private static void showTrainingPrecision(ComputationGraph vgg16, String classesNumber) throws IOException {
        File[] carTrackings = new File("CarTracking").listFiles();
        for (File carTracking : carTrackings) {
            if (carTracking.getName().contains(classesNumber)) {
                DataSetIterator dataSetIterator = ImageUtils.createDataSetIterator(carTracking,
                        Integer.parseInt(classesNumber), 64);
                Evaluation eval = vgg16.evaluate(dataSetIterator);
                System.out.println(eval.stats());
            }
        }
    }

    public static TestResult test(ComputationGraph vgg16, String model) throws IOException {
        Map<File, List<INDArray>> map = buildEmbeddings(vgg16);

        Set<Map.Entry<File, List<INDArray>>> entries = map.entrySet();
        int wrongPredictionsWithOtherClasses = 0;
        int wrongPredictionsInsideOneClassOrdered = 0;
        int wrongPredictionsInsideOneClassNotOrdered = 0;

        for (Map.Entry<File, List<INDArray>> entry : entries) {
            wrongPredictionsWithOtherClasses += compareWithOtherClasses(entries, entry);
            wrongPredictionsInsideOneClassOrdered += compareInsideOneClassOrdered(entry);
            wrongPredictionsInsideOneClassNotOrdered += compareInsideOneClassNotOrdered(entry);
        }
        return new TestResult(
                model,
                wrongPredictionsInsideOneClassOrdered,
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
    private static Map<File, List<INDArray>> buildEmbeddings(ComputationGraph model) throws IOException {
        File[] folders = new File(TEST_DATA).listFiles();
        Map<File, List<INDArray>> map = new HashMap<>();


        for (File folder : folders) {
            File[] carsInOneClass = folder.listFiles();

            Arrays.sort(carsInOneClass, Comparator.comparing(File::getName));

            Map<File, INDArray> fileEmbeddings = new TreeMap<>(Comparator.comparing(File::getName));
            Stream.of(carsInOneClass).forEach(inOneClass -> {
                try {
                    INDArray embeddings = getEmbeddings(model,
                            inOneClass);
                    fileEmbeddings.put(inOneClass, embeddings);
                } catch (IOException e) {
                    e.printStackTrace();
                }
            });

            map.put(folder, new ArrayList<>(fileEmbeddings.values()));
        }

        return map;
    }

    private static INDArray getEmbeddings(ComputationGraph vgg16, File image) throws IOException {
        INDArray indArray = LOADER.asMatrix(image);
        IMAGE_PRE_PROCESSOR.preProcess(indArray);
        Map<String, INDArray> stringINDArrayMap = vgg16.feedForward(indArray,false);
        INDArray embeddings = stringINDArrayMap.get("embeddings");
        return embeddings;
    }

    @AllArgsConstructor
    @Data
    static class TestResult {
        String model;
        int wrongPredictionsInsideOneClassSequentially;
        int wrongPredictionsInsideOneClassNotSequentially;
        int wrongPredictionsWithOtherClasses;
        int total;
    }

}
