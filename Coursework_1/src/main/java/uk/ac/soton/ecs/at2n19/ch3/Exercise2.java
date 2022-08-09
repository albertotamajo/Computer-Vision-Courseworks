package uk.ac.soton.ecs.at2n19.ch3;


import org.openimaj.image.DisplayUtilities;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.MBFImage;
import org.openimaj.image.pixel.ConnectedComponent;
import org.openimaj.image.segmentation.FelzenszwalbHuttenlocherSegmenter;
import org.openimaj.image.segmentation.SegmentationUtilities;

import java.io.IOException;
import java.net.URL;
import java.util.List;

/*
    The segmentation algorithm we just implemented can work reasonably well, but is rather naïve. OpenIMAJ contains an
    implementation of a popular segmentation algorithm called the FelzenszwalbHuttenlocherSegmenter.

    Try using the FelzenszwalbHuttenlocherSegmenter for yourself and see how it compares to the basic segmentation
    algorithm we implemented. You can use the SegmentationUtilities.renderSegments() static method to draw the connected
    components produced by the segmenter.
*/
public class Exercise2 {
    // Visually, the Felzenszwalb and Huttenlocher segmentation algorithm seems to achieve a better segmentation
    // performance in comparison to the solution illustrated in the tutorial. Besides, the Felzenszwalb and Huttenlocher
    // algorithm segments an image without receiving as input the number of clusters to be found.
    public static void main(String[] args) throws IOException {

        // Load image from URL
        MBFImage input = ImageUtilities.readMBF(new URL("http://static.openimaj.org/media/tutorial/sinaface.jpg"));
        // Create an instance of the segmentation algorithm described in "Efficient Graph-Based Image Segmentation"
        FelzenszwalbHuttenlocherSegmenter<MBFImage> segmenter = new FelzenszwalbHuttenlocherSegmenter<>();
        // Segment the image using FelzenszwalbHuttenlocherSegmenter
        List<ConnectedComponent> components = segmenter.segment(input);
        // Render the components to the image with randomly assigned colours
        MBFImage segmentedInput = SegmentationUtilities.renderSegments(input,components);
        // Display the image
        DisplayUtilities.display(segmentedInput);
    }
}
