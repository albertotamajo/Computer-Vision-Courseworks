package uk.ac.soton.ecs.at2n19.coursework3;

import org.openimaj.feature.DoubleFV;
import org.openimaj.feature.DoubleFVComparison;
import org.openimaj.image.FImage;
import org.openimaj.ml.annotation.basic.KNNAnnotator;


/**
 * K-nearest-neighbour classifier using the “tiny image” feature
 */
public class KNearestNeighborTinyImages extends KNNAnnotator<FImage, String, DoubleFV> {

    /**
     * Construct classifier with the given number of neighbours and the crop and resolution widths used for the process
     * of feature extraction
     * @param k number of neighbours
     * @param cropWidth width of the image cropped about the centre
     * @param resWidth width to be used to resize the cropped image
     */
    public KNearestNeighborTinyImages(int k, int cropWidth, int resWidth) {
        super(new TinyImageExtractor(cropWidth, resWidth), DoubleFVComparison.EUCLIDEAN,k);
    }
}
