package ramo.klevis.ml.tracking.data;

import ramo.klevis.ml.tracking.VideoPlayer;
import ramo.klevis.ml.tracking.yolo.Speed;

/**
 * Created by klevis.ramo on 10/19/2018
 */

public class ProduceDataFromVideo {
    public static void main(String[] args) throws Exception {
        VideoPlayer videoPlayer = new VideoPlayer();
        videoPlayer.startRealTimeVideoDetection("CarTracking/videoSample.mp4", "", Speed.MEDIUM,
                true, true);

    }
}
