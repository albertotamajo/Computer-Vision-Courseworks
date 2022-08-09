package uk.ac.soton.ecs.at2n19.ch5;

import org.openimaj.feature.local.list.LocalFeatureList;
import org.openimaj.feature.local.matcher.BasicMatcher;
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
import org.openimaj.math.geometry.transforms.estimation.RobustAffineTransformEstimator;
import org.openimaj.math.model.fit.RANSAC;

import java.io.IOException;
import java.net.URL;

public class Tutorial {
    // It is clear that finding matches between a query and a target image using the RANSAC model fitter for affine
    // transforms leads to a better result compared to a basic matcher. Such a result was expected as the RANSAC model
    // fitter for Affine transforms  is a robust model estimator and as such it is capable of determining which matches
    // are inliers and which ones are outliers with a larger degree of certainty.
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
        // Construct basic matcher that is capable of matching each SIFT descriptor in the query image with a SIFT
        // descriptor in the target image. Each SIFT descriptor X in the query image is matched with a SIFT descriptor Y
        // in the target image if Y is the closest descriptor to X and this distance is at most 80% of the distance
        // between X and its second closest SIFT descriptor Z in the target image. This method was proposed by David
        // Lowe
        LocalFeatureMatcher<Keypoint> matcher = new BasicMatcher<>(80);
        // Set the features that represent the database to match queries against
        matcher.setModelFeatures(queryKeypoints);
        // Find matches between the query image's SIFT descriptors and the target image's SIFT descriptors
        matcher.findMatches(targetKeypoints);
        // Draw the resulting matches between the query image and the target image
        MBFImage basicMatches = MatchingUtilities.drawMatches(query, target, matcher.getMatches(), RGBColour.RED);
        // Display the image showing the matches between the query image and the target image
        DisplayUtilities.display(basicMatches);

        // Create a RANSAC model fitter configured to find Affine Transforms
        RobustAffineTransformEstimator modelFitter = new RobustAffineTransformEstimator(50.0, 1500,
                new RANSAC.PercentageInliersStoppingCondition(0.5));
        // Create matcher that given an internal matcher and a model fitter finds which matches given by the internal
        // matcher are consistent with respect to the model. The internal matcher matches is a faster version of the
        // basic matcher used above.
        matcher = new ConsistentLocalFeatureMatcher2d<Keypoint>(
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
        // Draw a polygon around the estimated location of the query image within the target image
        target.drawShape(
                query.getBounds().transform(modelFitter.getModel().getTransform().inverse()), 3,RGBColour.BLUE);
        // Display the target image with such polygon
        DisplayUtilities.display(target);
    }
}
