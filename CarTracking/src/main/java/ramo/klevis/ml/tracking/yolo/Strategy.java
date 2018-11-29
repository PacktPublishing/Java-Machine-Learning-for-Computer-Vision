package ramo.klevis.ml.tracking.yolo;

public enum Strategy {

    IoU_PLUS_ENCODINGS("Use IoU and Encodings"),
    ONLY_IoU("Use Only IoU"),
    ONLY_ENCODINGS("Use Only Encodings");

    private String description;

    Strategy(String description) {
        this.description = description;
    }

    @Override
    public String toString() {
        return description;
    }
}
