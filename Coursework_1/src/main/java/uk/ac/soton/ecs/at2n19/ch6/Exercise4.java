package uk.ac.soton.ecs.at2n19.ch6;

import org.openimaj.data.dataset.MapBackedDataset;
import org.openimaj.image.DisplayUtilities;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.MBFImage;
import org.openimaj.image.dataset.BingImageDataset;
import org.openimaj.util.api.auth.DefaultTokenFactory;
import org.openimaj.util.api.auth.common.BingAPIToken;
import java.util.Map;

/*
    The MapBackedDataset class provides a concrete implementation of a GroupedDataset. See if you can use the static
    MapBackedDataset.of method to construct a grouped dataset of images of some famous people. Use a BingImageDataset
    to get the images of each person.
 */
public class Exercise4 {
    public static void main(String[] args) {

        BingAPIToken bingToken = DefaultTokenFactory.get(BingAPIToken.class);
        // Create an image dataset containing 10 images of Tom Cruise from the Bing search API.
        BingImageDataset<MBFImage> tomCruise = BingImageDataset.create(ImageUtilities.MBFIMAGE_READER,
                bingToken, "Tom Cruise", 10);
        // Create an image dataset containing 10 images of The Weeknd from the Bing search API.
        BingImageDataset<MBFImage> theWeeknd = BingImageDataset.create(ImageUtilities.MBFIMAGE_READER,
                bingToken, "The Weeknd", 10);
        // Create an image dataset containing 10 images of Will Smith from the Bing search API.
        BingImageDataset<MBFImage> willSmith = BingImageDataset.create(ImageUtilities.MBFIMAGE_READER,
                bingToken, "Will Smith", 10);
        // Create an image dataset containing 10 images of Post Malone from the Bing search API.
        BingImageDataset<MBFImage> postMalone = BingImageDataset.create(ImageUtilities.MBFIMAGE_READER,
                bingToken, "Post Malone", 10);

        // Construct a map backed dataset from the above Bing image datasets
        MapBackedDataset<String, BingImageDataset<MBFImage>, MBFImage> dataset = MapBackedDataset.of(tomCruise,
                theWeeknd, willSmith, postMalone);
        // Display all the images of each celebrity in a window
        for (Map.Entry<String, BingImageDataset<MBFImage>> entry : dataset.entrySet()) {
            DisplayUtilities.display(entry.getKey(), entry.getValue());
        }
    }
}
