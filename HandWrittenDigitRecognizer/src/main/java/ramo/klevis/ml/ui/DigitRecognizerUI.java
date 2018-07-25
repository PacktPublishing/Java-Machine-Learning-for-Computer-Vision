package ramo.klevis.ml.ui;

import lombok.extern.slf4j.Slf4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import ramo.klevis.ml.ImageOperationHelper;
import ramo.klevis.ml.LabeledImage;
import ramo.klevis.ml.network.neural.DigitRecognizerNeuralNetwork;
import ramo.klevis.ml.network.neural.DigitRecognizerConvolutionalNeuralNetwork;

import javax.swing.*;
import javax.swing.plaf.FontUIResource;
import java.awt.*;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
import java.awt.image.BufferedImage;
import java.util.concurrent.Executors;

@Slf4j
public class DigitRecognizerUI {

    private final static Logger LOGGER = LoggerFactory.getLogger(DigitRecognizerUI.class);

    private static final int FRAME_WIDTH = 1200;
    private static final int FRAME_HEIGHT = 628;
    private final DigitRecognizerNeuralNetwork neuralNetwork = new DigitRecognizerNeuralNetwork();
    private final DigitRecognizerConvolutionalNeuralNetwork convolutionalNeuralNetwork = new DigitRecognizerConvolutionalNeuralNetwork();

    private DrawArea drawArea;
    private JFrame mainFrame;
    private JPanel mainPanel;
    private JPanel drawAndDigitPredictionPanel;
    private JPanel resultPanel;

    public DigitRecognizerUI() throws Exception {
        UIManager.setLookAndFeel(UIManager.getSystemLookAndFeelClassName());
        UIManager.put("Button.font", new FontUIResource(new Font("Dialog", Font.BOLD, 18)));
        UIManager.put("ProgressBar.font", new FontUIResource(new Font("Dialog", Font.BOLD, 18)));
        neuralNetwork.init();
    }

    public void initUI() {
        // create main frame
        mainFrame = createMainFrame();

        mainPanel = new JPanel();
        mainPanel.setLayout(new BorderLayout());

        drawAndDigitPredictionPanel = new JPanel(new GridLayout());
        addActionPanel();
        addDrawAreaAndPredictionArea();
        mainPanel.add(drawAndDigitPredictionPanel, BorderLayout.CENTER);

        addSignature();

        mainFrame.add(mainPanel, BorderLayout.CENTER);
        mainFrame.setVisible(true);

    }

    private void addActionPanel() {
        JButton recognizeNN = new JButton("Recognize Digit With Simple NN");
        recognizeNN.addActionListener(e -> {
            LabeledImage labeledImage = getSelectedImage();
            int predict = neuralNetwork.predict(labeledImage);
            showDigitPredection(predict);

        });

        JButton recognizeCNN = new JButton("Recognize Digit With Convolutional NN");
        recognizeCNN.addActionListener(e -> {
            LabeledImage labeledImage = getSelectedImage();
            int predict = convolutionalNeuralNetwork.predict(labeledImage);
            showDigitPredection(predict);
        });

        JButton clear = new JButton("Clear");
        clear.addActionListener(e -> {
            drawArea.setImage(null);
            drawArea.repaint();
            drawAndDigitPredictionPanel.updateUI();
        });
        JPanel actionPanel = new JPanel(new GridLayout(8, 1));
        addTrainButton(actionPanel);
        actionPanel.add(recognizeNN);
        actionPanel.add(recognizeCNN);
        actionPanel.add(clear);
        drawAndDigitPredictionPanel.add(actionPanel);
    }

    private void showDigitPredection(int predict) {
        JLabel predictNumber = new JLabel("" + predict);
        predictNumber.setForeground(Color.RED);
        predictNumber.setFont(new Font("SansSerif", Font.BOLD, 128));
        resultPanel.removeAll();
        resultPanel.add(predictNumber);
        resultPanel.updateUI();
    }

    private LabeledImage getSelectedImage() {
        Image drawImage = drawArea.getImage();
        BufferedImage sbi = ImageOperationHelper.toBufferedImage(drawImage);
        Image scaled = ImageOperationHelper.scaleImage(sbi);
        BufferedImage scaledBuffered = ImageOperationHelper.toBufferedImage(scaled);
        double[] scaledPixels = ImageOperationHelper.transformRGBToGrayScale(scaledBuffered);
        return new LabeledImage(0, scaledPixels);
    }

    private void addDrawAreaAndPredictionArea() {

        drawArea = new DrawArea();

        drawAndDigitPredictionPanel.add(drawArea);
        resultPanel = new JPanel();
        resultPanel.setLayout(new GridBagLayout());
        drawAndDigitPredictionPanel.add(resultPanel);
    }

    private void addTrainButton(JPanel panel) {
        JButton trainNN = new JButton("Train Neural Network");
        trainNN.addActionListener(e -> {

            int i = JOptionPane.showConfirmDialog(mainFrame, "Are you sure this may take some time to train?");

            if (i == JOptionPane.OK_OPTION) {
                ProgressBar progressBar = new ProgressBar(mainFrame);
                SwingUtilities.invokeLater(() -> progressBar.showProgressBar("Training may take one or two minutes..."));
                Executors.newCachedThreadPool().submit(() -> {
                    try {
                        LOGGER.info("Start training Neural Network");
                        neuralNetwork.train();
                        LOGGER.info("Neural Network Training ended");
                    } catch (Exception e1) {
                        LOGGER.error("", e1);
                        throw new RuntimeException(e1);
                    } finally {
                        progressBar.setVisible(false);
                    }
                });
            }
        });
        panel.add(trainNN);
    }

    private JFrame createMainFrame() {
        JFrame mainFrame = new JFrame();
        mainFrame.setTitle("Digit Recognizer");
        mainFrame.setDefaultCloseOperation(WindowConstants.DISPOSE_ON_CLOSE);
        mainFrame.setSize(FRAME_WIDTH, FRAME_HEIGHT);
        mainFrame.setLocationRelativeTo(null);
        mainFrame.addWindowListener(new WindowAdapter() {
            @Override
            public void windowClosed(WindowEvent e) {
                System.exit(0);
            }
        });
        ImageIcon imageIcon = new ImageIcon("icon.png");
        mainFrame.setIconImage(imageIcon.getImage());

        return mainFrame;
    }

    private void addSignature() {
        JLabel signature = new JLabel("ramok.tech", SwingConstants.CENTER);
        signature.setFont(new Font(Font.SANS_SERIF, Font.ITALIC, 20));
        signature.setForeground(Color.BLUE);
        mainPanel.add(signature, BorderLayout.SOUTH);
    }

}