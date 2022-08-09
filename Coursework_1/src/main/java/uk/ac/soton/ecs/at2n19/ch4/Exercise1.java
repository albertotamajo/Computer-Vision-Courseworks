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
    Which images are most similar? Does that match with what you expect if you look at the images? Can you make the
    application display the two most similar images that are not the same?
 */
public class Exercise1 {

    // The first and second image are the most similar with respect to the distribution of pixel colours
    // according to the EUCLIDEAN DISTANCE between their multidimensional histograms. In other words, the
    // multidimensional histograms of the first and second image are separated by the lowest euclidean distance in
    // comparison to all the other pairs of different multidimensional histograms.
    // Such a result meets my expectations as the first and second image have similar colours (red-ish shades) while the
    // third image contains dark and blue shades.
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
        // Empty list of doubles used to store the euclidean distances between each different pair of images' multi dimensional histograms
        List<Double> distances = new ArrayList<>();
        // Compute and add to the list the euclidean distance between each different pair of the images' multi dimensional histograms
        for( int i = 0; i < histograms.size(); i++ ) {
            for( int j = i+1; j < histograms.size(); j++ ) {
                double distance = histograms.get(i).compare( histograms.get(j), DoubleFVComparison.EUCLIDEAN );
                distances.add(distance);
            }
        }
        // Index of the lowest euclidean distance
        int index = distances.indexOf(distances.stream().min(Comparator.comparingDouble(Double::valueOf)).get());

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
