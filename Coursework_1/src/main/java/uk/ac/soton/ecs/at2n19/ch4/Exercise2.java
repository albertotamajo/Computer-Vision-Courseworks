package uk.ac.soton.ecs.at2n19.ch4;

import org.openimaj.feature.DoubleFVComparison;
import org.openimaj.image.DisplayUtilities;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.pixel.statistics.HistogramModel;
import org.openimaj.math.statistics.distribution.MultidimensionalHistogram;

import java.io.IOException;
import java.net.URL;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

/*
    What happens when you use a different comparison measure (such as DoubleFVComparison.INTERSECTION)?
 */
public class Exercise2 {
    // Using the INTERSECTION comparison measure leads to the same result obtained using the EUCLIDEAN DISTANCE
    // comparison measure. Therefore, according to the INTERSECTION comparison measure, the first and second image
    // are the most similar with respect to the distribution of pixel colours. It is important to notice that while in
    // the EUCLIDEAN DISTANCE measure the most similar pair of different pictures is the one that achieves the lowest
    // value, in the INTERSECTION comparison measure the most similar pair of different pictures is the one that
    // achieves the highest value. This follows from the fact that the higher is the similarity between two histograms,
    // the higher is the value of their intersection.
    // Once again, the result obtained using the INTERSECTION comparison measure meets my expectations because the
    // multidimensional histograms of the first and second image are the most similar (the images have similar colours)
    // and so their intersection achieves the highest intersection comparison measure value.
    public static void main(String[] args) throws IOException {

        // List of URLs for images
        URL[] imageURLs = new URL[] {
                new URL( "http://openimaj.org/tutorial/figs/hist1.jpg" ),
                new URL( "http://openimaj.org/tutorial/figs/hist2.jpg" ),
                new URL( "http://openimaj.org/tutorial/figs/hist3.jpg" )
        };
        // Empty list of multi dimensional histograms
        List<MultidimensionalHistogram> histograms = new ArrayList<>();
        // Create an instance capable of generating a multi dimensional histogram contaiing 64 bins for a given image
        HistogramModel model = new HistogramModel(4, 4, 4);
        // Generate and add to a list each image's multi dimensional histogram
        for( URL u : imageURLs ) {
            model.estimateModel(ImageUtilities.readMBF(u));
            histograms.add( model.histogram.clone());
        }
        // Empty list of doubles used to store the intersection between each different pair of images' multi dimensional histograms
        List<Double> distances = new ArrayList<>();
        // Compute and add to the list the intersection between each different pair of the images' multi dimensional histograms
        for( int i = 0; i < histograms.size(); i++ ) {
            for( int j = i+1; j < histograms.size(); j++ ) {
                double distance = histograms.get(i).compare( histograms.get(j),  DoubleFVComparison.INTERSECTION);
                System.out.println(distance);
                distances.add(distance);
            }
        }
        // Index of the maximum intersection
        int index = distances.indexOf(distances.stream().max(Comparator.comparingDouble(Double::valueOf)).get());

        // The first and second image are the most similar. Thus, display them
        if (index == 0){
            DisplayUtilities.display("First and second image are the most similar", ImageUtilities.readMBF(imageURLs[0]), ImageUtilities.readMBF(imageURLs[1]));
        } else if ( index == 1){ // The first and third image are the most similar. Thus, display them
            DisplayUtilities.display("First and third image are the most similar", ImageUtilities.readMBF(imageURLs[0]), ImageUtilities.readMBF(imageURLs[2]));
        } else { // The second and third image are the most similar. Thus, display them
            DisplayUtilities.display("Second and third image are the most similar", ImageUtilities.readMBF(imageURLs[1]), ImageUtilities.readMBF(imageURLs[2]));
        }

    }
}
