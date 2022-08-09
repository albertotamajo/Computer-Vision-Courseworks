package uk.ac.soton.ecs.at2n19.ch13;

import org.apache.commons.vfs2.FileSystemException;
import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.ListDataset;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.experiment.dataset.split.GroupedRandomSplitter;
import org.openimaj.experiment.dataset.util.DatasetAdaptors;
import org.openimaj.feature.DoubleFV;
import org.openimaj.feature.DoubleFVComparison;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.model.EigenImages;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

/*
    In the original Eigenfaces paper, a variant of nearest-neighbour classification was used that incorporated a
    distance threshold. If the distance between the query face and closest database face was greater than a threshold,
    then an unknown result would be returned, rather than just returning the label of the closest person.
    Can you alter your code to include such a threshold? What is a good value for the threshold?
 */
public class Exercise3 {
    // In order to find the best threshold an empirical approach is used.
    // Such approach computes the average number of correct, incorrect and unknown guesses over the thresholds ranging
    // from 6 to 20. The average number of correct, incorrect and unknown guesses for a given threshold is computed
    // over 10 repetitions. The results obtained are the following:

    // Average results for threshold 6 is: corrects 58 - incorrects 0 - unknowns 141  | f 2.43
    // Average results for threshold 7 is: corrects 85 - incorrects 0 - unknowns 114  | f 1.31
    // Average results for threshold 8 is: corrects 115 - incorrects 0 - unknowns 84  | f 0.73
    // Average results for threshold 9 is: corrects 147 - incorrects 0 - unknowns 51  | f 0.34
    // Average results for threshold 10 is: corrects 161 - incorrects 1 - unknowns 36 | f 0.24
    // Average results for threshold 11 is: corrects 171 - incorrects 4 - unknowns 24 | f 0.21
    // Average results for threshold 12 is: corrects 181 - incorrects 5 - unknowns 13 | f 0.15
    // Average results for threshold 13 is: corrects 184 - incorrects 9 - unknowns 5  | f 0.17
    // Average results for threshold 14 is: corrects 187 - incorrects 12 - unknowns 0 | f 0.19
    // Average results for threshold 15 is: corrects 187 - incorrects 12 - unknowns 0 | f 0.19
    // Average results for threshold 16 is: corrects 187 - incorrects 12 - unknowns 0 | f 0.19
    // Average results for threshold 17 is: corrects 186 - incorrects 13 - unknowns 0 | f 0.20
    // Average results for threshold 18 is: corrects 188 - incorrects 11 - unknowns 0 | f 0.17
    // Average results for threshold 19 is: corrects 189 - incorrects 11 - unknowns 0 | f 0.17
    // Average results for threshold 20 is: corrects 189 - incorrects 11 - unknowns 0 | f 0.17

    // However, what is the best threshold among the ones listed above? That depends on what evaluation measure is used.
    // For example, the best threshold could be the largest one that does not produce incorrect results. In this case,
    // the best threshold would be 9. However, on the other hand, the best threshold could be the one that does not
    // produce too many incorrects and at the same time does produce many correct guesses.
    // A general function that allows to construct evaluation metrics for this problem is the following:
    //  f(corrects, incorrects, unknowns; weight) = (unknowns + weight * incorrects) / corrects
    //
    // The threshold that achieves the lowest function value is the best threshold. The weight value in the function
    // allows to decide how much importance the number of incorrect guesses should have in the process of decision taking.
    // In the table above the weight used is 3 and the best threshold according to this evaluation measure
    // turns out to be 12.


    public static void main(String[] args) throws FileSystemException {

        // Compute the average number of correct, incorrect and unknown guesses for each threshold ranging from 6 to 20
        // and print them out
        for (int i = 6; i <= 20; i++){
            int max_reps = 10;
            int[] avgScore = new int[3];
            // Compute the average number of correct, incorrect and unknown guesses for each threshold by averaging
            // over 10 repetitions
            for(int r = 0; r < max_reps; r++){
                int[] score = training_test_loop(i);
                avgScore[0] += score[0];
                avgScore[1] += score[1];
                avgScore[2] += score[2];
            }
            avgScore[0] /= max_reps;
            avgScore[1] /= max_reps;
            avgScore[2] /= max_reps;
            System.out.printf("Average results for threshold %d is: corrects %d - incorrects %d - unknowns %d \n",
                    i, avgScore[0], avgScore[1], avgScore[2]);
        }
    }


    static int[] training_test_loop(double threshold) throws FileSystemException {
        // Load dataset of faces
        VFSGroupDataset<FImage> dataset = new VFSGroupDataset<>("zip:http://datasets.openimaj.org/att_faces" +
                ".zip", ImageUtilities.FIMAGE_READER);
        // number of training samples for each group
        int nTraining = 5;
        // number of testing samples for each group
        int nTesting = 5;
        // Split the dataset into two halves; one for training and the other one for testing
        GroupedRandomSplitter<String, FImage> splits = new GroupedRandomSplitter<>(dataset, nTraining, 0, nTesting);
        // Training dataset
        GroupedDataset<String, ListDataset<FImage>, FImage> training = splits.getTrainingDataset();
        // Test dataset
        GroupedDataset<String, ListDataset<FImage>, FImage> testing = splits.getTestDataset();
        // Create a list view of the training dataset
        List<FImage> basisImages = DatasetAdaptors.asList(training);
        // Feature dimension
        int nEigenvectors = 100;
        // Create an instance of EigenImages which performs dimensionality reduction on an image using PCA
        EigenImages eigen = new EigenImages(nEigenvectors);
        // Train with the basis images
        eigen.train(basisImages);


        // Create a database of features from the training images
        Map<String, DoubleFV[]> features = new HashMap<>();
        for (final String person : training.getGroups()) {
            final DoubleFV[] fvs = new DoubleFV[nTraining];

            for (int i = 0; i < nTraining; i++) {
                final FImage face = training.get(person).get(i);
                fvs[i] = eigen.extractFeature(face);
            }
            features.put(person, fvs);
        }

        // Loop over all the testing images and for each of them find the database feature with the smallest distance
        // (i.e. Euclidean distance) and return the identifier of the corresponding person unless the smallest distance
        // is larger than the threshold. In that case, do not give any prediction
        // A list containing the number of correct, incorrect and unknown guesses is returned
        int correct = 0, incorrect = 0, unknowns = 0;
        for (String truePerson : testing.getGroups()) {
            for (FImage face : testing.get(truePerson)) {
                DoubleFV testFeature = eigen.extractFeature(face);

                String bestPerson = null;
                double minDistance = Double.MAX_VALUE;
                for (final String person : features.keySet()) {
                    for (final DoubleFV fv : features.get(person)) {
                        double distance = fv.compare(testFeature, DoubleFVComparison.EUCLIDEAN);

                        if (distance < minDistance) {
                            minDistance = distance;
                            bestPerson = person;
                        }
                    }
                }
                // if the min distance is larger than the threshold then do not give a prediction
                if(minDistance > threshold){
                    bestPerson = "unknown";
                }

                if (truePerson.equals(bestPerson))  // correctly guessed
                    correct++;
                else if (bestPerson.equals("unknown")) // no prediction
                    unknowns++;
                else  // wrongly guessed
                    incorrect++;

            }
        }
        return new int[]{correct, incorrect, unknowns};
    }
}