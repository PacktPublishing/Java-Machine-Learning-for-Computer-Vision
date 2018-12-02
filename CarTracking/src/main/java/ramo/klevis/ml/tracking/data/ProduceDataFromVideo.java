package ramo.klevis.ml.tracking.data;

import ramo.klevis.ml.tracking.VideoPlayer;
import ramo.klevis.ml.tracking.yolo.Strategy;

/**
 * Created by klevis.ramo on 10/19/2018
 */

public class ProduceDataFromVideo {
    public static void main(String[] args) throws Exception {
        VideoPlayer videoPlayer = new VideoPlayer();
        videoPlayer.startRealTimeVideoDetection("CarTracking/src/main/resources/videoSample.mp4", "",
                true, 0.85, "611_epoch_data_e512_b512_1120.zip", null);

    }
}
