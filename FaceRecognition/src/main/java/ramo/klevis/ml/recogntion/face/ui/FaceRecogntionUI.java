package ramo.klevis.ml.recogntion.face.ui;


import lombok.extern.slf4j.Slf4j;
import ramo.klevis.ml.recogntion.face.FaceRecognition;
import ramo.klevis.ml.ui.ImagePanel;

import javax.swing.*;
import javax.swing.plaf.FontUIResource;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
import java.io.File;
import java.io.IOException;
import java.util.Objects;

/**
 * Created by Klevis Ramo
 */
@Slf4j
public class FaceRecogntionUI {
    public static final String FACE_RECOGNITION_SRC_MAIN_RESOURCES = "FaceRecognition/src/main/resources/images";
    private static final String BASE_PATH = "FaceRecognition/src/main/resources/";
    private JFrame mainFrame;
    private JPanel mainPanel;
    private static final int FRAME_WIDTH = 1024;
    private static final int FRAME_HEIGHT = 420;
    private ImagePanel sourceImagePanel;
    private FaceRecognition faceRecognition;
    private File selectedFile;
    private JTextField memberNameField;
    private final Font sansSerifBold = new Font("SansSerif", Font.BOLD, 28);
    private JPanel membersPhotosPanel;
    private JScrollPane scrollMembersPhotos;
    private JLabel whoIsLabel;

    public FaceRecogntionUI() throws Exception {
        UIManager.setLookAndFeel(UIManager.getSystemLookAndFeelClassName());
        UIManager.put("Button.font", new FontUIResource(new Font("Dialog", Font.BOLD, 18)));
        UIManager.put("ProgressBar.font", new FontUIResource(new Font("Dialog", Font.BOLD, 18)));

    }

    public void initUI() throws Exception {

        faceRecognition = new FaceRecognition();
        faceRecognition.loadModel();
        // create main frame
        mainFrame = createMainFrame();

        mainPanel = new JPanel(new BorderLayout());

        JButton chooseButton = new JButton("Choose Face Image");
        chooseButton.addActionListener(e -> {
            chooseFileAction();
            mainPanel.updateUI();
        });

        JButton whoIsButton = new JButton("Who Is?");
        whoIsButton.addActionListener(event -> {
            try {
                String whoIs = faceRecognition.whoIs(selectedFile.getAbsolutePath());
                whoIsLabel.setFont(sansSerifBold);
                if (!whoIs.contains("Unknown")) {
                    whoIsLabel.setForeground(Color.GREEN.darker());
                } else {
                    whoIsLabel.setForeground(Color.RED.darker());
                }
                whoIsLabel.setText(whoIs);
                mainPanel.updateUI();
            } catch (IOException e) {
                log.error("", e);
                throw new RuntimeException(e);
            }
        });

        JButton registerNewMemberButton = new JButton("Register New Member");
        registerNewMemberButton.addActionListener(new AbstractAction() {
            @Override
            public void actionPerformed(ActionEvent event) {
                try {
                    addPhoto(selectedFile.getAbsolutePath(), memberNameField.getText());
                    membersPhotosPanel.updateUI();
                    scrollMembersPhotos.updateUI();
                    mainPanel.updateUI();
                } catch (IOException e) {
                    log.error("", e);
                    throw new RuntimeException(e);
                }
            }
        });

        fillMainPanel(chooseButton, whoIsButton, registerNewMemberButton);

        mainPanel.updateUI();

        mainFrame.add(mainPanel, BorderLayout.CENTER);
        addSignature();
        mainFrame.setVisible(true);

    }

    private void fillMainPanel(JButton chooseButton, JButton predictButton, Component registerNewMemberButton) throws IOException {

        GridLayout layout = new GridLayout(1, 4);
        layout.setHgap(1);
        JPanel panelRegister = new JPanel(layout);
        memberNameField = new JTextField();

        panelRegister.add(registerNewMemberButton);
        panelRegister.add(memberNameField);
        panelRegister.add(chooseButton);
        panelRegister.add(predictButton);
        mainPanel.add(panelRegister, BorderLayout.NORTH);

        membersPhotosPanel = new JPanel(new GridLayout(1, 15));
        scrollMembersPhotos = new JScrollPane(membersPhotosPanel);
        mainPanel.add(scrollMembersPhotos, BorderLayout.SOUTH);

        File[] files = new File(BASE_PATH + "/images").listFiles();
        for (File file : Objects.requireNonNull(files)) {
            File[] images = file.listFiles();
            addPhoto(Objects.requireNonNull(images)[0].getAbsolutePath());

        }
        sourceImagePanel = new ImagePanel(150, 150);
        JPanel jPanel = new JPanel(new GridLayout(1, 2));
        jPanel.add(sourceImagePanel);
        whoIsLabel = new JLabel("?");
        whoIsLabel.setFont(sansSerifBold);
        whoIsLabel.setForeground(Color.BLUE.darker());
        jPanel.add(whoIsLabel);
        mainPanel.add(jPanel, BorderLayout.CENTER);
        mainPanel.updateUI();

    }

    private void addPhoto(String path) throws IOException {
        addPhoto(path, null);
    }

    private void addPhoto(String path, String name) throws IOException {
        ImagePanel imagePanel = new ImagePanel(150, 150, new File(path), name);
        faceRecognition.registerNewMember(imagePanel.getTitle(), new File(imagePanel.getFilePath()).getAbsolutePath());
        membersPhotosPanel.add(imagePanel);
    }


    public void chooseFileAction() {
        JFileChooser chooser = new JFileChooser();
        chooser.setCurrentDirectory(new File(new File(FACE_RECOGNITION_SRC_MAIN_RESOURCES).getAbsolutePath()));
        int action = chooser.showOpenDialog(null);
        if (action == JFileChooser.APPROVE_OPTION) {
            try {
                selectedFile = chooser.getSelectedFile();
                sourceImagePanel.setImage(selectedFile.getAbsolutePath());
            } catch (IOException e) {
                log.error("", e);
                throw new RuntimeException(e);
            }
        }
    }

    private JFrame createMainFrame() {
        JFrame mainFrame = new JFrame();
        mainFrame.setTitle("Face Recognizer");
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

    private void addSignature() {
        JLabel signature = new JLabel("ramok.tech", SwingConstants.CENTER);
        signature.setFont(new Font(Font.SANS_SERIF, Font.ITALIC, 20));
        signature.setForeground(Color.BLUE);
        mainFrame.add(signature, BorderLayout.SOUTH);
    }
}
