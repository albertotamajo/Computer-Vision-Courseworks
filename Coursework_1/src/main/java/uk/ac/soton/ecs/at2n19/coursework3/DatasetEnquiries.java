package uk.ac.soton.ecs.at2n19.coursework3;

import org.apache.commons.vfs2.FileSystemException;
import org.openimaj.data.dataset.ListDataset;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.image.FImage;

import java.util.ArrayList;
import java.util.List;

/**
 * Class that contains a static method to get the minimum height and width across all the images in both the
 * training and test datasets
 */
public class DatasetEnquiries {

    /**
     * Get the minimum height and width across all the images in both the training and test datasets
     * @return a list containing the minimum width and height across all images
     * @throws FileSystemException
     */
    // Min width: 203
    // Min height: 200
    public static List<Integer> getMinWidthHeight() throws FileSystemException {

        VFSGroupDataset<FImage> trainDataset = Dataset.getTrainDataset();
//        VFSGroupDataset<FImage> testDataset = Dataset.getTestDataset();
        int minHeight = Integer.MAX_VALUE;
        int minWidth = Integer.MAX_VALUE;
        for(ListDataset<FImage> list : trainDataset.values()){
            for(FImage image : list){
                int width = image.getWidth();
                int height = image.getHeight();
                if (width < minWidth)
                    minWidth = width;
                if (height < minHeight)
                    minHeight = height;
            }
        }

//        for(ListDataset<FImage> list : testDataset.values()){
//            for(FImage image : list){
//                int width = image.getWidth();
//                int height = image.getHeight();
//                if (width < minWidth)
//                    minWidth = width;
//                if (height < minHeight)
//                    minHeight = height;
//            }
//        }

        List<Integer> list = new ArrayList<>();
        list.add(minWidth);
        list.add(minHeight);
        return list;

    }
}
