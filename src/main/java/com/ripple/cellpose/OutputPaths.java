package com.ripple.cellpose;

import java.io.File;

class OutputPaths {

    static String baseName(File file) {
        String name = file.getName();
        int dot = name.lastIndexOf('.');
        return dot > 0 ? name.substring(0, dot) : name;
    }

    static File getImageOutputDir(File imageFile) {
        File dir = new File(new File(imageFile.getParentFile(), "outputs"), baseName(imageFile));
        if (!dir.exists()) {
            dir.mkdirs();
        }
        return dir;
    }

    static File getOutputsRootDir(File anyImageFile) {
        File dir = new File(anyImageFile.getParentFile(), "outputs");
        if (!dir.exists()) {
            dir.mkdirs();
        }
        return dir;
    }

    static File buildMaskOutputFile(File imageFile) {
        return new File(getImageOutputDir(imageFile), baseName(imageFile) + "_mask.png");
    }

    static File buildMaskPropertiesOutputFile(File imageFile) {
        return new File(getImageOutputDir(imageFile), baseName(imageFile) + "_mask_properties.csv");
    }

    static File buildClassificationCsvFile(File imageFile) {
        return new File(getImageOutputDir(imageFile), baseName(imageFile) + "_classification.csv");
    }

    private OutputPaths() {}
}
