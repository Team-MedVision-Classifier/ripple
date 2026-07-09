package com.ripple.cellpose;

import java.awt.image.BufferedImage;

class SegmentationResult {
    final BufferedImage rawMask;
    final boolean noMasksFound;

    SegmentationResult(BufferedImage rawMask, boolean noMasksFound) {
        this.rawMask = rawMask;
        this.noMasksFound = noMasksFound;
    }
}
