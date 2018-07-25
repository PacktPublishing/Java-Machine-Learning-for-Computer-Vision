package ramo.klevis.ml.recogntion.face;

import lombok.extern.slf4j.Slf4j;
import ramo.klevis.ml.recogntion.face.ui.FaceRecogntionUI;
import ramo.klevis.ml.ui.ProgressBar;

import javax.swing.*;
import java.util.concurrent.Executors;


/**
 * Created by Klevis Ramo
 */
@Slf4j
public class RunFaceRecognition {

    public static void main(String[] args) throws Exception {

        JFrame mainFrame = new JFrame();
        ProgressBar progressBar = new ProgressBar(mainFrame, true);
        progressBar.showProgressBar("Loading model, this make take few moments");
        FaceRecogntionUI faceRecogntionUi = new FaceRecogntionUI();
        Executors.newCachedThreadPool().submit(() -> {
            try {
                faceRecogntionUi.initUI();
            } catch (Exception e) {
                log.error("Failed to start", e);
                throw new RuntimeException(e);
            } finally {
                progressBar.setVisible(false);
                mainFrame.dispose();
            }
        });

    }

}
