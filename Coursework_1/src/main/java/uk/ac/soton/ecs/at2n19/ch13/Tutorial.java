package uk.ac.soton.ecs.at2n19.ch13;

import org.apache.commons.vfs2.FileSystemException;
import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.ListDataset;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.experiment.dataset.split.GroupedRandomSplitter;
import org.openimaj.experiment.dataset.util.DatasetAdaptors;
import org.openimaj.feature.DoubleFV;
import org.openimaj.feature.DoubleFVComparison;
import org.openimaj.image.DisplayUtilities;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.model.EigenImages;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class Tutorial {
    public static void main(String[] args) throws FileSystemException {
        // Load dataset of faces
        VFSGroupDataset<FImage> dataset = new VFSGroupDataset<>("zip:http://datasets.openimaj.org/att_faces" +
                ".zip", ImageUtilities.FIMAGE_READER);
        // number of training samples per group
        int nTraining = 5;
        // number of testing samples per group
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

        // Display the first 11 principal components
        List<FImage> eigenFaces = new ArrayList<>();
        for (int i = 0; i < 12; i++) {
            eigenFaces.add(eigen.visualisePC(i));
        }
        DisplayUtilities.display("EigenFaces", eigenFaces);

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
        // The overall accuracy is printed out.
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

                System.out.println("Actual: " + truePerson + "\tguess: " + bestPerson);

                if (truePerson.equals(bestPerson))
                    correct++;
                else
                    incorrect++;
            }
        }

        System.out.println("Accuracy: " + (correct / (correct + incorrect)));

    }
}
