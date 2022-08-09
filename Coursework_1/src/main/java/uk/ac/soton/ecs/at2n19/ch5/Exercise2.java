package uk.ac.soton.ecs.at2n19.ch5;

import org.openimaj.feature.local.list.LocalFeatureList;
import org.openimaj.feature.local.matcher.FastBasicKeypointMatcher;
import org.openimaj.feature.local.matcher.LocalFeatureMatcher;
import org.openimaj.feature.local.matcher.MatchingUtilities;
import org.openimaj.feature.local.matcher.consistent.ConsistentLocalFeatureMatcher2d;
import org.openimaj.image.DisplayUtilities;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.MBFImage;
import org.openimaj.image.colour.RGBColour;
import org.openimaj.image.feature.local.engine.DoGSIFTEngine;
import org.openimaj.image.feature.local.keypoints.Keypoint;
import org.openimaj.math.geometry.transforms.HomographyRefinement;
import org.openimaj.math.geometry.transforms.estimation.RobustHomographyEstimator;
import org.openimaj.math.model.fit.RANSAC;

import java.io.IOException;
import java.net.URL;

/*
    Experiment with different models (such as a HomographyModel) in the consistent matcher.
    The RobustHomographyEstimator helper class can be used to construct an object that fits the HomographyModel model.
    You can also experiment with an alternative robust fitting algorithm to RANSAC called Least Median of Squares
    (LMedS) through the RobustHomographyEstimator.
 */
public class Exercise2 {
    // The performance of the RANSAC model fitter for Homographies and the LMeds model fitter for Homographies with 70%
    // proportion of outliers are comparable
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
        // Create a RANSAC model fitter configured to find Homographies
        RobustHomographyEstimator modelFitter = new RobustHomographyEstimator(50.0, 1500,
                new RANSAC.PercentageInliersStoppingCondition(0.5), HomographyRefinement.NONE);
        // Create matcher that given an internal matcher and a model fitter finds which matches given by the internal
        // matcher are consistent with respect to the model
        LocalFeatureMatcher<Keypoint> matcher = new ConsistentLocalFeatureMatcher2d<Keypoint>(
                new FastBasicKeypointMatcher<Keypoint>(8), modelFitter);
        // Set the features that represent the database to match queries against
        matcher.setModelFeatures(queryKeypoints);
        // Find matches between the query image's SIFT descriptors and the target image's SIFT descriptors
        matcher.findMatches(targetKeypoints);
        // Draw the resulting matches between the query image and the target image
        MBFImage consistentMatches = MatchingUtilities.drawMatches(query, target, matcher.getMatches(),
                RGBColour.RED);
        // Display the image showing the matches between the query image and the target image
        DisplayUtilities.display(consistentMatches);



        // Create a LMedS model fitter configured to find homographies
        modelFitter = new RobustHomographyEstimator(0.7, HomographyRefinement.NONE);
        // Create matcher that given an internal matcher and a model fitter finds which matches given by the internal
        // matcher are consistent with respect to the model
        matcher = new ConsistentLocalFeatureMatcher2d<Keypoint>(new FastBasicKeypointMatcher<Keypoint>(8), modelFitter);
        // Set the features that represent the database to match queries against
        matcher.setModelFeatures(queryKeypoints);
        // Find matches between the query image's keypoints and the target image's keypoints
        matcher.findMatches(targetKeypoints);
        // Draw the resulting matches between the query image and the target image
        consistentMatches = MatchingUtilities.drawMatches(query, target, matcher.getMatches(), RGBColour.RED);
        // Display the image showing the matches between the query image and the target image
        DisplayUtilities.display(consistentMatches);

    }
}
