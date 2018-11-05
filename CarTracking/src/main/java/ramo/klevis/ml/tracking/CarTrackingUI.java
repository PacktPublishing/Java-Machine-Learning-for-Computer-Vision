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

/**
 * Created by Klevis Ramo.
 */
public class CarTrackingUI {

    public final static AtomicInteger atomicInteger = new AtomicInteger();
    private static final int FRAME_WIDTH = 750;
    private static final int FRAME_HEIGHT = 220;
    private static final Font FONT = new Font("Dialog", Font.BOLD, 18);
    private static final Font FONT_ITALIC = new Font("Dialog", Font.ITALIC, 18);
    private static final String AUTONOMOUS_DRIVING_RAMOK_TECH = "Autonomous Driving(ramok.tech)";
    private JFrame mainFrame;
    private JPanel mainPanel;
    private File selectedFile = new File("AutonomousDriving/src/main/resources/videoSample.mp4");
    private VideoPlayer videoPlayer;
    private ProgressBar progressBar;
    private JRadioButton yolo;

    public void initUI() throws Exception {
        adjustLookAndFeel();
        mainFrame = createMainFrame();
        mainPanel = new JPanel(new FlowLayout(FlowLayout.CENTER));
        yoloChooser();
        final JComboBox<Speed> choose = speedChooser();

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
                            String absolutePath = selectedFile.getAbsolutePath();
                            System.out.println("absolutePath = " + absolutePath);
                            videoPlayer.startRealTimeVideoDetection(absolutePath,
                                    "1" + atomicInteger.incrementAndGet(),
                                    (Speed) choose.getSelectedItem(),
                                    yolo.isSelected(), false);
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
        addSignature();

        mainFrame.add(mainPanel, BorderLayout.CENTER);
        mainFrame.setVisible(true);

       /* File[] videos = new File("AutonomousDriving/src/main/resources/videos").listFiles();
        for (File file : videos) {

            new VideoPlayer().startRealTimeVideoDetection(file.getAbsolutePath(),
                    file.getName() + atomicInteger.incrementAndGet(), Speed.MEDIUM, true);
        }*/

    }

    private void yoloChooser() {
        JPanel panel = new JPanel(new GridLayout(0, 2));
        Border border = BorderFactory.createTitledBorder("Choose Yolo");
        ((TitledBorder) border).setTitleFont(FONT_ITALIC);
        panel.setBorder(border);
        ButtonGroup group = new ButtonGroup();
        yolo = new JRadioButton("Load Real Yolo");
        yolo.setFont(FONT);
        group.add(yolo);
        panel.add(yolo);
        JRadioButton tinyYolo = new JRadioButton("Tiny Yolo");
        tinyYolo.setSelected(true);
        panel.add(tinyYolo);
        group.add(tinyYolo);
        tinyYolo.setFont(FONT);
        mainPanel.add(panel);
    }

    private void adjustLookAndFeel() throws ClassNotFoundException, InstantiationException, IllegalAccessException, UnsupportedLookAndFeelException {
        UIManager.setLookAndFeel(UIManager.getSystemLookAndFeelClassName());
        UIManager.put("Button.font", new FontUIResource(FONT));
        UIManager.put("ComboBox.font", new FontUIResource(FONT_ITALIC));
        UIManager.put("ProgressBar.font", new FontUIResource(FONT));
    }

    @NotNull
    private JComboBox<Speed> speedChooser() {
        JPanel panel = new JPanel();
        TitledBorder speed = BorderFactory.createTitledBorder("Speed");
        speed.setTitleFont(FONT_ITALIC);
        speed.setTitleColor(Color.BLUE.darker());
        panel.setBorder(speed);
        mainPanel.add(panel);
        JComboBox<Speed> choose = new JComboBox<>();
        choose.setBackground(Color.BLUE.darker());
        choose.setForeground(Color.BLUE.darker());
        choose.addItem(Speed.FAST);
        choose.addItem(Speed.MEDIUM);
        choose.addItem(Speed.SLOW);
        choose.setSelectedIndex(2);
        panel.add(choose);
        return choose;
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
