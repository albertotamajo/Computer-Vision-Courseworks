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
    The number of images used for training can have a big effect in the performance of your recogniser. Try reducing the
    number of training images (keep the number of testing images fixed at 5). What do you observe?
 */
public class Exercise2 {

    // As the the training set size decreases, the test accuracy decreases as well.
    // The main method below computes the average test accuracy for each training size
    // ranging from 1 to 5. The average test accuracy for each training size is computed
    // by averaging the results obtained over 10 repetitions.
    // The accuracy results are listed below:
    //      Average accuracy for training size 1 is: 0.685000
    //      Average accuracy for training size 2 is: 0.827000
    //      Average accuracy for training size 3 is: 0.875000
    //      Average accuracy for training size 4 is: 0.911500
    //      Average accuracy for training size 5 is: 0.941000
    public static void main(String[] args) throws FileSystemException {

        // array which stores the average test accuracy for each training size ranging from 1 to 5
        double[] accuracies = new double[5];
        // Compute the average test accuracy for each training size ranging from 1 to 5 and store
        // the results in the accuracies array
        for (int i = 0; i < 5; i++){
            int max_reps = 10;
            double accuracy = 0;
            // Compute the average accuracy for each training size by averaging over 10 repetitions
            for(int r = 0; r < max_reps; r++){
                accuracy += training_test_loop(i+1,5);
            }
            accuracies[i] = accuracy / max_reps;
        }
        // Print out the average accuracy for each training size
        for(int i = 0; i < 5; i++){
            System.out.printf("Average accuracy for training size %d is: %f\n", i+1, accuracies[i]);
        }

    }

    public static double training_test_loop(int nTraining, int nTesting) throws FileSystemException {
        // Load dataset of faces
        VFSGroupDataset<FImage> dataset = new VFSGroupDataset<>("zip:http://datasets.openimaj.org/att_faces" +
                ".zip", ImageUtilities.FIMAGE_READER);
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
        // (i.e. Euclidean distance) and return the identifier of the corresponding person.
        // The overall accuracy is returned
        double correct = 0, incorrect = 0;
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

                if (truePerson.equals(bestPerson))
                    correct++;
                else
                    incorrect++;
            }
        }

        return correct / (correct + incorrect);
    }
}
