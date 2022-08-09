package uk.ac.soton.ecs.at2n19.ch13;

import org.apache.commons.vfs2.FileSystemException;
import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.ListDataset;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.experiment.dataset.split.GroupedRandomSplitter;
import org.openimaj.experiment.dataset.util.DatasetAdaptors;
import org.openimaj.feature.DoubleFV;
import org.openimaj.image.DisplayUtilities;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.model.EigenImages;

import java.util.*;

/*
    An interesting property of the features extracted by the Eigenfaces algorithm (specifically from the PCA process)
    is that it's possible to reconstruct an estimate of the original image from the feature. Try doing this by building
    a PCA basis as described above, and then extract the feature of a randomly selected face from the test-set.
    Use the EigenImages#reconstruct() to convert the feature back into an image and display it. You will need to
    normalise the image (FImage#normalise()) to ensure it displays correctly as the reconstruction might give pixel
    values bigger than 1 or smaller than 0.
 */
public class Exercise1 {
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


        // Convert the set of person identifiers into a list
        List<String> groups = new ArrayList<>(testing.getGroups());
        // Shuffle the list in-place
        Collections.shuffle(groups);
        // Get a random image from the person identifier located at index 0 of the shuffled list
        FImage image = testing.get(groups.get(0)).getRandomInstance();
        // Extract features from the image
        DoubleFV imageFeature = eigen.extractFeature(image);
        // Display the real image and the image reconstructed from the extracted features
        DisplayUtilities.display("Real vs Reconstructed",image, eigen.reconstruct(imageFeature).normalise());

    }
}
