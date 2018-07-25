package ramo.klevis.ml.vg16;

import lombok.AllArgsConstructor;
import lombok.Data;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

/**
 * Created by Klevis Ramo
 */
@Data
@AllArgsConstructor
public class NeuralNetworkTrainingData {
    private final DataSetIterator trainIterator;
    private final DataSetIterator devIterator;
    private final DataSetIterator testIterator;

}
