package ramo.klevis.ml;

import lombok.extern.slf4j.Slf4j;
import org.apache.commons.io.FileUtils;
import ramo.klevis.ml.ui.ProgressBar;
import ramo.klevis.ml.ui.CatVsDogUI;

import javax.swing.*;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
import java.io.File;
import java.io.IOException;
import java.net.URL;
import java.util.concurrent.Executors;
import java.util.zip.Adler32;

import static ramo.klevis.ml.vg16.CatVsDogRecognition.TRAINED_PATH_MODEL;

/**
 * Created by Klevis Ramo
 */
@Slf4j
public class RunCatVsDogRecognizer {

    private static final String MODEL_URL = "https://dl.dropboxusercontent.com/s/djmh91tk1bca4hz/RunEpoch_class_2_soft_10_32_1800.zip?dl=0";

    public static void main(String[] args) throws Exception {
        downloadModelForFirstTime();

        JFrame mainFrame = new JFrame();
        ProgressBar progressBar = new ProgressBar(mainFrame, true);
        progressBar.showProgressBar("Loading model, this make take few moments");
        CatVsDogUI catVsDogUi = new CatVsDogUI();
        Executors.newCachedThreadPool().submit(() -> {
            try {
                catVsDogUi.initUI();
            } catch (Exception e) {
                log.error("Failed to start",e);
                throw new RuntimeException(e);
            } finally {
                progressBar.setVisible(false);
                mainFrame.dispose();
            }
        });

    }

    private static void downloadModelForFirstTime() throws IOException {
        JFrame mainFrame = new JFrame();
        mainFrame.addWindowListener(new WindowAdapter() {
            @Override
            public void windowClosing(WindowEvent e) {
                System.exit(0);
            }
        });
        ProgressBar progressBar = new ProgressBar(mainFrame, false);
        File model = new File(TRAINED_PATH_MODEL);
        if (!model.exists() || FileUtils.checksum(model, new Adler32()).getValue() != 3082129141L) {
            model.delete();
            progressBar.showProgressBar("Downloading model for the first time 500MB!");
            URL modelURL = new URL(MODEL_URL);

            try {
                FileUtils.copyURLToFile(modelURL, model);
            } catch (IOException e) {
                JOptionPane.showMessageDialog(null, "Failed to download model");
                throw new RuntimeException(e);
            } finally {
                progressBar.setVisible(false);
                mainFrame.dispose();
            }

        }
    }
}
