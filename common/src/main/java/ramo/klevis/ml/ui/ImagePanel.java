package ramo.klevis.ml.ui;

import org.apache.commons.lang3.StringUtils;

import javax.imageio.ImageIO;
import javax.swing.*;
import javax.swing.border.TitledBorder;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;

/**
 * Created by Klevis Ramo
 */
public class ImagePanel extends JPanel {

    public static final int DEFAULT_WIDTH = 400;
    public static final int DEFAULT_HEIGHT = 400;

    public final int scaleWidth;
    public int scaleHeight;
    private String title;

    private Image img;

    private InputStream imageStream;
    private String filePath;

    public ImagePanel(int scaleWidth, int scaleHeight) throws IOException {
        this.scaleWidth = scaleWidth;
        this.scaleHeight = scaleHeight;
        showDefault();
    }


    public ImagePanel() throws IOException {
        this.scaleWidth = DEFAULT_WIDTH;
        this.scaleHeight = DEFAULT_HEIGHT;
        showDefault();
    }

    public ImagePanel(int scaleWidth, int scaleHeight, File imageFile, String title) throws IOException {
        this(scaleWidth, scaleHeight);
        setImage(imageFile.getAbsolutePath());
        if (StringUtils.isBlank(title)) {
            title = imageFile.getName().replaceAll("_", "")
                    .replace(".jpg", "")
                    .replace(".png", "")
                    .replaceAll("[0-9]", "");
        }
        TitledBorder titledBorder = BorderFactory.createTitledBorder(title);
        titledBorder.setTitleColor(Color.RED.darker());
        setBorder(titledBorder);
        this.title = title;

    }

    private void imageToStream(String filePath) throws FileNotFoundException {
        this.filePath = filePath;
        imageStream = new FileInputStream(this.filePath);
    }

    public void showDefault() throws IOException {
        String showDefaultImage = getDefaultImage();
        setImage("common/src/main/resources/" + showDefaultImage);
    }


    public void setImage(String file) throws IOException {
        imageToStream(file);
        BufferedImage bufferedImage = ImageIO.read(imageStream);
        Image scaledInstance = bufferedImage.getScaledInstance(scaleWidth, scaleHeight, Image.SCALE_DEFAULT);
        setImage(scaledInstance);
    }

    public void paintComponent(Graphics g) {
        g.drawImage(img, 0, 0, null);
    }

    private void setImage(Image img) {
        this.img = img;
        Dimension size = new Dimension(scaleWidth, scaleHeight);
        setPreferredSize(size);
        setMinimumSize(size);
        setMaximumSize(size);
        setSize(size);
        setLayout(null);
        repaint();
        updateUI();
    }

    private String getDefaultImage() {
        return "/placeholder.gif";
    }


    public String getTitle() {
        return title;
    }

    public String getFilePath() {
        return filePath;
    }
}