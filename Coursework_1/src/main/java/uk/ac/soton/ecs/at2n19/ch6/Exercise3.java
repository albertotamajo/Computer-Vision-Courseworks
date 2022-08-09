package uk.ac.soton.ecs.at2n19.ch6;

import org.openimaj.image.DisplayUtilities;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.MBFImage;
import org.openimaj.image.dataset.BingImageDataset;
import org.openimaj.util.api.auth.DefaultTokenFactory;
import org.openimaj.util.api.auth.common.BingAPIToken;

/*
    The BingImageDataset class allows you to create a dataset of images by performing a search using the Bing search
    engine. The BingImageDataset class works in a similar way to the FlickrImageDataset described above. Try it out!
 */
public class Exercise3 {
    public static void main(String[] args) {

        BingAPIToken bingToken = DefaultTokenFactory.get(BingAPIToken.class);
        // Create an image dataset containing 10 cats from the Bing search API.
        BingImageDataset<MBFImage> cats = BingImageDataset.create(ImageUtilities.MBFIMAGE_READER,
                bingToken, "cat", 10);
        // Display the dataset in a grid-like fashion
        DisplayUtilities.display("Cats", cats);
    }
}
