package uk.ac.soton.ecs.at2n19.ch5;

import org.openimaj.feature.local.list.LocalFeatureList;
import org.openimaj.feature.local.matcher.*;
import org.openimaj.image.DisplayUtilities;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.MBFImage;
import org.openimaj.image.colour.RGBColour;
import org.openimaj.image.feature.local.engine.DoGSIFTEngine;
import org.openimaj.image.feature.local.keypoints.Keypoint;
import java.io.IOException;
import java.net.URL;

/*
    Experiment with different matchers; try the BasicTwoWayMatcher for example.
 */
public class Exercise1 {
    // The basic two way matcher outperforms the basic matcher used in the Tutorial. This result meets my expectations
    // as the basic two way matcher uses a matching conditions that is more rigid than the one used in the basic matcher
    // and so many more mismatches are discarded. On the other hand, the basic two way matcher does not nearly reach the
    // performance of the RANSAC model fitter for Affine transforms. Such a result was also expected as the RANSAC model
    // fitter for Affine transforms  is a robust model estimator and as such it is capable of determining which matches
    // are inliers and which ones are outliers with a larger degree of certainty.

    // It seems that the voting key point matcher slightly outperforms the basic two way matcher but it is also no near
    // the performance reached by the RANSAC model fitter for Affine transforms.
    public static void main(String[] args) throws IOException {

        // Load query image from URL
        MBFImage query = ImageUtilities.readMBF(new URL("http://static.openimaj.org/media/tutorial/query.jpg"));
        // Load target image from URL
        MBFImage target = ImageUtilities.readMBF(new URL("http://static.openimaj.org/media/tutorial/target.jpg"));
        // Construct difference-of-Gaussian feature detector with SIFT descriptor engine
        // Such engine detects difference-of-Gaussian interest points in an image and describes them
        // with a SIFT descriptor
        DoGSIFTEngine engine = new DoGSIFTEngine();
        // Find features in the query image
        LocalFeatureList<Keypoint> queryKeypoints = engine.findFeatures(query.flatten());
        // Find features in the target image
        LocalFeatureList<Keypoint> targetKeypoints = engine.findFeatures(target.flatten());
        // Construct basic two way matcher that is capable of matching each SIFT descriptor in the query image with a
        // SIFT descriptor in the target image. It uses the Euclidean distance to find matches. A match between a SIFT
        // descriptor X of the query image and a SIFT descriptor Y of the target image occurs only when X matches with Y
        // and Y matches with X. Thus, each match is a two-way match.
        LocalFeatureMatcher<Keypoint> matcher = new BasicTwoWayMatcher<>();
        // Set the features that represent the database to match queries against
        matcher.setModelFeatures(queryKeypoints);
        // Find matches between the query image's SIFT descriptors and the target image's SIFT descriptors
        matcher.findMatches(targetKeypoints);
        // Draw the resulting matches between the query image and the target image
        MBFImage basicMatches = MatchingUtilities.drawMatches(query, target, matcher.getMatches(), RGBColour.RED);
        // Display the image showing the matches between the query image and the target image
        DisplayUtilities.display(basicMatches);


        // Construct voting basic matcher which rejects matches with no local support
        matcher = new VotingKeypointMatcher<>(8);
        // Set the features that represent the database to match queries against
        matcher.setModelFeatures(queryKeypoints);
        // Find matches between the query image's keypoints and the target image's keypoints
        matcher.findMatches(targetKeypoints);
        // Draw the resulting matches between the query image and the target image
        basicMatches = MatchingUtilities.drawMatches(query, target, matcher.getMatches(), RGBColour.RED);
        // Display the image showing the matches between the query image and the target image
        DisplayUtilities.display(basicMatches);
    }
}
