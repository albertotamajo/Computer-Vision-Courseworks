package uk.ac.soton.ecs.at2n19.ch4;

import org.openimaj.feature.DoubleFVComparison;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.pixel.statistics.HistogramModel;
import org.openimaj.math.statistics.distribution.MultidimensionalHistogram;

import java.io.IOException;
import java.net.URL;
import java.util.ArrayList;
import java.util.List;

public class Tutorial {
    public static void main(String[] args) throws IOException {

        // List of URLs for images
        URL[] imageURLs = new URL[] {
                new URL( "http://openimaj.org/tutorial/figs/hist1.jpg" ),
                new URL( "http://openimaj.org/tutorial/figs/hist2.jpg" ),
                new URL( "http://openimaj.org/tutorial/figs/hist3.jpg" )
        };
        // Empty list of multi dimensional histograms
        List<MultidimensionalHistogram> histograms = new ArrayList<>();
        // Create an instance capable of generating a multi dimensional histogram containing 64 bins for a given image
        HistogramModel model = new HistogramModel(4, 4, 4);
        // Generate and add to a list each image's multi dimensional histogram
        for( URL u : imageURLs ) {
            model.estimateModel(ImageUtilities.readMBF(u));
            histograms.add( model.histogram.clone());
        }
        // Compute and print out the euclidean distance between each pair of images' multi dimensional histograms
        for( int i = 0; i < histograms.size(); i++ ) {
            for( int j = i; j < histograms.size(); j++ ) {
                double distance = histograms.get(i).compare( histograms.get(j), DoubleFVComparison.EUCLIDEAN );
                System.out.printf("Distance between image %d and image %d: %f\n", i,j,distance);
            }
        }
    }
}
