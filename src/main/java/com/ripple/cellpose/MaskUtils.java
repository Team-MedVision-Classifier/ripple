package com.ripple.cellpose;

import java.awt.Color;
import java.awt.image.BufferedImage;
import java.awt.image.Raster;
import java.io.File;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import javax.imageio.ImageIO;

class MaskUtils {

    static int getMaskLabel(Raster raster, int x, int y) {
        return raster.getNumBands() > 0 ? raster.getSample(x, y, 0) : 0;
    }

    static boolean isBoundaryPixel(Raster raster, int width, int height, int x, int y, int label) {
        if (x == 0 || y == 0 || x == width - 1 || y == height - 1) {
            return true;
        }
        return getMaskLabel(raster, x - 1, y) != label
            || getMaskLabel(raster, x + 1, y) != label
            || getMaskLabel(raster, x, y - 1) != label
            || getMaskLabel(raster, x, y + 1) != label;
    }

    static BufferedImage createColoredMaskFromLabels(BufferedImage labelImage, int[] numCellsOut, Color[] colorMap) {
        int width = labelImage.getWidth();
        int height = labelImage.getHeight();
        BufferedImage colored = new BufferedImage(width, height, BufferedImage.TYPE_INT_ARGB);
        HashSet<Integer> labels = new HashSet<>();
        Raster raster = labelImage.getRaster();
        int bands = raster.getNumBands();

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int label = bands >= 1 ? raster.getSample(x, y, 0) : 0;
                if (label <= 0) {
                    colored.setRGB(x, y, 0);
                    continue;
                }
                labels.add(label);
                Color color = colorMap[Math.floorMod(label, colorMap.length)];
                int argb = (255 << 24) | (color.getRed() << 16) | (color.getGreen() << 8) | color.getBlue();
                colored.setRGB(x, y, argb);
            }
        }

        if (numCellsOut != null && numCellsOut.length > 0) {
            numCellsOut[0] = labels.size();
        }
        return colored;
    }

    static Map<Integer, int[]> computeCellCentroidsFromMask(File maskFile) {
        Map<Integer, int[]> centroids = new HashMap<>();
        try {
            BufferedImage maskImg = ImageIO.read(maskFile);
            if (maskImg == null) {
                return centroids;
            }
            int width = maskImg.getWidth();
            int height = maskImg.getHeight();
            Raster raster = maskImg.getRaster();
            Map<Integer, long[]> accum = new HashMap<>();

            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    int label = getMaskLabel(raster, x, y);
                    if (label <= 0) {
                        continue;
                    }
                    long[] sums = accum.computeIfAbsent(label, k -> new long[3]);
                    sums[0] += x;
                    sums[1] += y;
                    sums[2]++;
                }
            }

            for (Map.Entry<Integer, long[]> entry : accum.entrySet()) {
                long[] sums = entry.getValue();
                centroids.put(entry.getKey(), new int[]{(int) (sums[0] / sums[2]), (int) (sums[1] / sums[2])});
            }
            System.out.println("[Cellpose] Computed centroids for " + centroids.size() + " cells");
        } catch (Exception e) {
            System.err.println("[Cellpose] Failed to compute centroids: " + e.getMessage());
        }
        return centroids;
    }

    private MaskUtils() {}
}
