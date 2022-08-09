package uk.ac.soton.ecs.at2n19.coursework3;

import org.openimaj.feature.DoubleFV;
import org.openimaj.feature.FeatureExtractor;
import org.openimaj.image.FImage;
import org.openimaj.image.feature.FImage2DoubleFV;
import org.openimaj.image.processing.algorithm.MeanCenter;
import org.openimaj.image.processing.resize.ResizeProcessor;

/**
 * Crop each image to a square about the centre and then resize it to a small, fixed resolution.
 * Each pixel is packed into a vector by concatenating each row of the cropped image.
 */
public class TinyImageExtractor implements FeatureExtractor<DoubleFV, FImage> {

    private final int cropWidth;
    private final int resWidth;


    /**
     * Construct the extractor with the given crop and resolution widths
     * @param cropWidth width of the image cropped about the centre
     * @param resWidth width to be used to resize the cropped image
     */
    public TinyImageExtractor(int cropWidth, int resWidth) {
        this.cropWidth = cropWidth;
        this.resWidth = resWidth;
    }


    /**
     * Extract features from an image and return them.
     * The image is cropped about the center, mean centered and then normalised.
     * @param image image to extract features from
     * @return normalised vector produced by the concatenation of the cropped image's rows
     */
    @Override
    public DoubleFV extractFeature(FImage image) {
        // Crop image about the center and resize it
        FImage croppedImage = image.extractCenter(cropWidth, cropWidth).process(( new ResizeProcessor(resWidth,resWidth)));
        // Mean center the image
        croppedImage.processInplace(new MeanCenter());
        // Return a normalised vector produced by the concatenation of the cropped image's rows
        return new FImage2DoubleFV().extractFeature(croppedImage).normaliseFV(2);
    }
}
