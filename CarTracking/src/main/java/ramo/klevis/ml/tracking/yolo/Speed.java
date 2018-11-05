package ramo.klevis.ml.tracking.yolo;

public enum Speed {

    FAST("Real-Time but low accuracy", 224, 224, 7, 7),
    MEDIUM("Almost Real-time and medium accuracy", 416, 416, 13, 13),
    SLOW("Slowest but high accuracy", 608, 608, 19, 19);

    public final int width;
    public final int height;
    public final int gridWidth;
    public final int gridHeight;
    private final String name;

    Speed(String name, int width, int height, int gridWidth, int gridHeight) {

        this.name = name;
        this.width = width;
        this.height = height;
        this.gridWidth = gridWidth;
        this.gridHeight = gridHeight;
    }

    public String getName() {
        return name;
    }

    @Override
    public String toString() {
        return name;
    }
}
