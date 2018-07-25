package ramo.klevis.ml.ui;

import lombok.extern.slf4j.Slf4j;
import ramo.klevis.ml.EdgeDetection;

import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.*;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

import static ramo.klevis.ml.EdgeDetection.*;

/**
 * Created by Klevis Ramo
 */
@Slf4j
public class EdgeDetectionUI {

    private static final int FRAME_WIDTH = 1000;
    private static final int FRAME_HEIGHT = 600;
    private static final Font sansSerifBold = new Font("SansSerif", Font.BOLD, 22);
    private final ImagePanel sourceImage = new ImagePanel(500, 510);
    private final ImagePanel destImage = new ImagePanel(500, 510);
    private final JPanel mainPanel;
    private final EdgeDetection edgeDetection;

    public EdgeDetectionUI() throws IOException {

        edgeDetection = new EdgeDetection();
        JFrame mainFrame = createMainFrame();

        mainPanel = new JPanel(new GridLayout(1, 2));
        mainPanel.add(sourceImage);
        mainPanel.add(destImage);

        JPanel northPanel = fillNorthPanel();

        mainFrame.add(northPanel, BorderLayout.NORTH);
        mainFrame.add(mainPanel, BorderLayout.CENTER);
        mainFrame.setVisible(true);
    }

    private JPanel fillNorthPanel() {
        JButton chooseButton = new JButton("Choose Image");
        chooseButton.setFont(sansSerifBold);

        JPanel northPanel = new JPanel();
        JComboBox filterType = new JComboBox();
        filterType.addItem(HORIZONTAL_FILTER);
        filterType.addItem(VERTICAL_FILTER);

        filterType.addItem(SOBEL_FILTER_VERTICAL);
        filterType.addItem(SOBEL_FILTER_HORIZONTAL);

        filterType.addItem(SCHARR_FILTER_VETICAL);
        filterType.addItem(SCHARR_FILTER_HORIZONTAL);
        filterType.setFont(sansSerifBold);

        JButton detect = new JButton("Detect Edges");
        detect.setFont(sansSerifBold);

        northPanel.add(filterType);
        northPanel.add(chooseButton);
        northPanel.add(detect);

        chooseButton.addActionListener(event -> {
            JFileChooser chooser = new JFileChooser();
            chooser.setCurrentDirectory(new File("EdgeDetection/src/main/resources"));
            int action = chooser.showOpenDialog(null);
            if (action == JFileChooser.APPROVE_OPTION) {
                try {
                    sourceImage.setImage(chooser.getSelectedFile().getAbsolutePath());
                    mainPanel.updateUI();
                } catch (IOException e) {
                    log.error("", e);
                    throw new RuntimeException(e);
                }
            }
        });

        detect.addActionListener(event -> {
            try {
                BufferedImage bufferedImage = ImageIO.read(new File(sourceImage.getFilePath()));
                File convolvedFile = edgeDetection.detectEdges(bufferedImage, (String) filterType.getSelectedItem());
                destImage.setImage(convolvedFile.getAbsolutePath());
            } catch (IOException e) {
                log.error("", e);
                throw new RuntimeException(e);
            }
        });

        return northPanel;
    }

    private JFrame createMainFrame() {
        JFrame mainFrame = new JFrame();
        mainFrame.setDefaultCloseOperation(WindowConstants.DISPOSE_ON_CLOSE);
        mainFrame.setSize(FRAME_WIDTH, FRAME_HEIGHT);
        mainFrame.setLocationRelativeTo(null);
        mainFrame.addWindowListener(new WindowAdapter() {
            @Override
            public void windowClosed(WindowEvent e) {
                System.exit(0);
            }
        });
        return mainFrame;
    }

}
