package uk.ac.soton.ecs.at2n19.coursework3;

import org.openimaj.feature.FeatureExtractor;
import org.openimaj.feature.FeatureVector;
import org.openimaj.feature.local.list.LocalFeatureList;
import org.openimaj.feature.local.list.MemoryLocalFeatureList;
import org.openimaj.image.FImage;

/**
 * A window slides across the image and extracts features from those regions.
 * As the window slides across the image, a feature is extracted from the region inside the window and it is added to
 * the list of local features.
 * @param <T> feature vector extracted as the window slides
 */
public class SlidingWindowExtractor<T extends FeatureVector> implements FeatureExtractor<LocalFeatureList<SlidingWindowLocalFeature<T>>,FImage> {

    private final int w;
    private final int h;
    private final int stepX;
    private final int stepY;
    private final FeatureExtractor<T,FImage> featExtractor;


    /**
     * Construct the extractor with a window of the given width and height which slides across the image
     * according to the given steps along the x and y directions. As the window slides, a feature is extracted from the
     * region inside the window with the given feature extractor.
     * @param w width of the window
     * @param h height of the window
     * @param stepX step along the x direction
     * @param stepY step along the y direction
     * @param featExtractor feature extractor for the region inside the sliding window
     */
    public SlidingWindowExtractor(int w, int h, int stepX, int stepY, FeatureExtractor<T,FImage> featExtractor) {
        this.w = w;
        this.h = h;
        this.stepX = stepX;
        this.stepY = stepY;
        this.featExtractor = featExtractor;
    }


    /**
     * Extract features from an image and return them.
     * A window slides across the image and extracts features from those regions.
     * @param image image to extract features from
     * @return list of local features extracted by the sliding window
     */
    @Override
    public LocalFeatureList<SlidingWindowLocalFeature<T>> extractFeature(FImage image) {
        MemoryLocalFeatureList<SlidingWindowLocalFeature<T>> featList = new MemoryLocalFeatureList<>();
        // Slide the window across the image and meanwhile extract a feature from the region inside the window and
        // add it to the list of local features
        for (int y = 0; y + h -1 < image.getHeight(); y+=stepY){
            for (int x = 0; x + w - 1 < image.getWidth(); x+=stepX){
                FImage window = image.extractROI(x,y,w,h);
                T featVector = featExtractor.extractFeature(window);
                featList.add(new SlidingWindowLocalFeature<>(x,y,featVector));
            }
        }
        return featList;
    }
}
