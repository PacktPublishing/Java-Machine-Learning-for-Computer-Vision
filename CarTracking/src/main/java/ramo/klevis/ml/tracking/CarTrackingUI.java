package ramo.klevis.ml.tracking;

import org.jetbrains.annotations.NotNull;
import ramo.klevis.ml.tracking.yolo.Speed;
import ramo.klevis.ml.ui.ProgressBar;

import javax.swing.*;
import javax.swing.border.Border;
import javax.swing.border.TitledBorder;
import javax.swing.plaf.FontUIResource;
import java.awt.*;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
import java.io.File;
import java.io.IOException;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Stream;

/**
 * Created by Klevis Ramo.
 */
public class CarTrackingUI {

    public final static AtomicInteger atomicInteger = new AtomicInteger();
    private static final int FRAME_WIDTH = 550;
    private static final int FRAME_HEIGHT = 220;
    private static final Font FONT = new Font("Dialog", Font.BOLD, 18);
    private static final Font FONT_ITALIC = new Font("Dialog", Font.ITALIC, 18);
    private static final String AUTONOMOUS_DRIVING_RAMOK_TECH = "Car Tracking (ramok.tech)";
    private JFrame mainFrame;
    private JPanel mainPanel;
    private File selectedFile = new File("CarTracking/src/main/resources/videoSample.mp4");
    private VideoPlayer videoPlayer;
    private ProgressBar progressBar;
    private JComboBox<String> chooseCifar10Model;
    private JSpinner threshold;

    public void initUI() throws Exception {
        adjustLookAndFeel();
        mainFrame = createMainFrame();
        mainPanel = new JPanel(new FlowLayout(FlowLayout.CENTER));

        JPanel actionPanel = new JPanel();
        actionPanel.setBorder(BorderFactory.createTitledBorder(""));

        JButton chooseVideo = new JButton("Choose Video");
        chooseVideo.setBackground(Color.ORANGE.darker());
        chooseVideo.setForeground(Color.ORANGE.darker());
        chooseVideo.addActionListener(e -> chooseFileAction());
        actionPanel.add(chooseVideo);

        JButton start = new JButton("Start Detection!");
        start.setBackground(Color.GREEN.darker());
        start.setForeground(Color.GREEN.darker());

        start.addActionListener(e -> {
            progressBar = new ProgressBar(mainFrame);
            SwingUtilities.invokeLater(() -> progressBar.showProgressBar("Detecting video..."));
            Executors.newSingleThreadExecutor().submit(() -> {
                try {
                    videoPlayer = new VideoPlayer();
                    Runnable runnable1 = () -> {
                        try {
                            videoPlayer.startRealTimeVideoDetection(selectedFile.getAbsolutePath(),
                                    String.valueOf(atomicInteger.incrementAndGet()),
                                    false,
                                    (Double) threshold.getValue(),
                                    (String) chooseCifar10Model.getSelectedItem());
                        } catch (Exception e1) {
                            e1.printStackTrace();

                        }
                    };
                    new Thread(runnable1).start();
                } catch (Exception e1) {
                    throw new RuntimeException(e1);
                } finally {
                    progressBar.setVisible(false);
                }
            });
        });
        actionPanel.add(start);

        JButton stop = new JButton("Stop");
        stop.setBackground(Color.RED);
        stop.setForeground(Color.RED);
        stop.addActionListener(e -> {
            if (videoPlayer == null) {
                return;
            }
            try {
                videoPlayer.stop();
            } catch (IOException e1) {
                e1.printStackTrace();
            }
            progressBar.setVisible(false);
        });
        actionPanel.add(stop);

        mainPanel.add(actionPanel);

        chooseCifar10Model = new JComboBox<>();
        chooseCifar10Model.setForeground(Color.BLUE);
        Stream.of(new File("CarTracking/src/main/resources/models").listFiles())
                .forEach(e -> chooseCifar10Model.addItem(e.getName()));
        JLabel label = new JLabel("Cifar-10 model");
        label.setForeground(Color.BLUE);
        mainPanel.add(label);
        mainPanel.add(chooseCifar10Model);

        label = new JLabel("Threshold");
        label.setForeground(Color.DARK_GRAY);
        threshold = new JSpinner(new SpinnerNumberModel(0.9, 0.1, 2, 0.1));
        threshold.setFont(FONT_ITALIC);
        mainPanel.add(label);
        mainPanel.add(threshold);

        addSignature();

        mainFrame.add(mainPanel, BorderLayout.CENTER);
        mainFrame.setVisible(true);

    }

    private void adjustLookAndFeel() throws ClassNotFoundException, InstantiationException, IllegalAccessException, UnsupportedLookAndFeelException {
        UIManager.setLookAndFeel(UIManager.getSystemLookAndFeelClassName());
        UIManager.put("Button.font", new FontUIResource(FONT));
        UIManager.put("Label.font", new FontUIResource(FONT_ITALIC));
        UIManager.put("ComboBox.font", new FontUIResource(FONT_ITALIC));
        UIManager.put("ProgressBar.font", new FontUIResource(FONT));
    }

    public void chooseFileAction() {
        JFileChooser chooser = new JFileChooser();
        chooser.setCurrentDirectory(new File(new File("AutonomousDriving/src/main/resources").getAbsolutePath()));
        int action = chooser.showOpenDialog(null);
        if (action == JFileChooser.APPROVE_OPTION) {
            selectedFile = chooser.getSelectedFile();

        }
    }

    private JFrame createMainFrame() {
        JFrame mainFrame = new JFrame();
        mainFrame.setTitle("Autonomous Driving");
        mainFrame.setDefaultCloseOperation(WindowConstants.DISPOSE_ON_CLOSE);
        mainFrame.setSize(FRAME_WIDTH, FRAME_HEIGHT);
        mainFrame.setMaximumSize(new Dimension(FRAME_WIDTH, FRAME_HEIGHT));
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
        mainFrame.add(signature, BorderLayout.SOUTH);
    }
}
