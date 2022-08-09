package uk.ac.soton.ecs.at2n19.coursework3;

import org.apache.commons.vfs2.FileSystemException;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;

/**
 * Class that contains static methods to get the training and test datasets
 */
public class Dataset {

    /**
     * Get training dataset
     * @return training dataset
     * @throws FileSystemException if the dataset cannot be reached
     */
    public static VFSGroupDataset<FImage> getTrainDataset() throws FileSystemException {
        VFSGroupDataset<FImage> groupedDataset = new VFSGroupDataset<>("zip:http://comp3204.ecs.soton.ac.uk/cw/training.zip",
                ImageUtilities.FIMAGE_READER);
        groupedDataset.remove("training");
        return groupedDataset;
    }

    /**
     * Get test dataset
     * @return test dataset
     * @throws FileSystemException if the dataset cannot be reached
     */
    public static VFSListDataset<FImage> getTestDataset() throws FileSystemException {
        return new VFSListDataset<>("zip:http://comp3204.ecs.soton.ac.uk/cw/testing.zip", ImageUtilities.FIMAGE_READER);
    }
}
