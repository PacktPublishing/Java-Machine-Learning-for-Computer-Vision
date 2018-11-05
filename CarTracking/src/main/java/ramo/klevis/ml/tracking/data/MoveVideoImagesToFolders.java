package ramo.klevis.ml.tracking.data;

import org.jetbrains.annotations.NotNull;

import java.io.File;
import java.util.Optional;
import java.util.stream.Stream;

/**
 * Created by klevis.ramo on 10/20/2018
 */

public class MoveVideoImagesToFolders {

    private static final String PATHNAME = "CarTracking\\VideoFolder";
    private static final int OFFSET = 405;

    public static void main(String[] args) {

        File[] files = new File(
                PATHNAME).listFiles();

        for (File file : files) {
            String name = file.getName();
            int i = name.indexOf("(");
            String folderNumber = null;
            if (i != -1) {
                folderNumber = name.substring(0, name.indexOf(" "));
            } else {
                continue;
            }
            String folder = String.valueOf(Integer.parseInt(folderNumber) + OFFSET);
            File file1 = new File(PATHNAME + "\\" + folder);
            if (!file1.exists()) {
                file1.mkdir();
            }
            file.renameTo(new File(PATHNAME + "\\" + folder + "\\" + file.getName()));
        }
        files = new File(
                PATHNAME).listFiles();

        for (File file : files) {
            if (file.isDirectory()) {

                String name = file.getName();
                Optional<File> first = Stream.of(files).parallel().filter(f ->
                        f.getName().length() <= 8 &&
                                !f.isDirectory() &&
                                removeJpg(f).equals("" + (Integer.parseInt(name) - OFFSET))).findFirst();
                if (!first.isPresent()) {
                    continue;
                }
                File file1 = first.get();

                file1.renameTo(new File(file.getAbsolutePath() + "\\" + file1.getName()));
            }
        }

    }

    @NotNull
    private static String removeJpg(File file) {
        return file.getName().substring(0,
                file.getName().length() - 4);
    }
}

