package ramo.klevis.ml;

import lombok.extern.slf4j.Slf4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import ramo.klevis.ml.ui.DigitRecognizerUI;
import ramo.klevis.ml.ui.ProgressBar;

import javax.swing.*;
import java.util.concurrent.Executors;

/**
 * Created by Klevis Ramo
 */
@Slf4j
public class RunDigitRecognizer {

    private final static Logger LOGGER = LoggerFactory.getLogger(RunDigitRecognizer.class);
    private static final JFrame mainFrame = new JFrame();

    public static void main(String[] args) throws Exception {

        LOGGER.info("Application is starting ... ");

        ProgressBar progressBar = new ProgressBar(mainFrame, true);
        progressBar.showProgressBar("Collecting data this make take several seconds!");
        DigitRecognizerUI digitRecognizerUi = new DigitRecognizerUI();
        Executors.newCachedThreadPool().submit(()->{
            try {
                digitRecognizerUi.initUI();
            } finally {
                progressBar.setVisible(false);
                mainFrame.dispose();
            }
        });
    }
}
