package uk.ac.soton.ecs.at2n19.ch6;

import org.openimaj.data.dataset.ListDataset;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.image.DisplayUtilities;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.dataset.FlickrImageDataset;
import org.openimaj.util.api.auth.DefaultTokenFactory;
import org.openimaj.util.api.auth.common.FlickrAPIToken;

import java.util.Map;

public class Tutorial {
    public static void main(String[] args) throws Exception {

        // Create a list dataset backed by a directory containing images
        // The dataset contains grey-scale versions of the images contained in the directory
        ListDataset<FImage> images = new VFSListDataset<FImage>("C:\\Users\\tamaj\\OneDrive\\University of Southampton\\" +
                "University-Of-Southampton-CourseWork\\Third_Year\\ComputerVision\\Coursework_1\\" +
                "OpenIMAJ-Tutorials\\images", ImageUtilities.FIMAGE_READER);
        // Print out the number of images in the list dataset
        System.out.println(images.size());
        // Display a random image contained in the list dataset
        DisplayUtilities.display(images.getRandomInstance(), "A random image from the dataset");
        // Display all images inside the list dataset in the same window in a grid-like fashion
        DisplayUtilities.display("My images", images);
        // Create a list dataset backed by a zip file hosted in a web-server
        VFSListDataset<FImage> faces = new VFSListDataset<>("zip:http://datasets.openimaj.org/att_faces.zip",
                ImageUtilities.FIMAGE_READER);
        // Display all images inside the list dataset in the same window in a grid-like fashion
        DisplayUtilities.display("ATT faces", faces);
        // Create a group dataset backed by a zip file hosted in a web-server
        VFSGroupDataset<FImage> groupedFaces = new VFSGroupDataset<FImage>(
                "zip:http://datasets.openimaj.org/att_faces.zip", ImageUtilities.FIMAGE_READER);
        // Display all the images from each category in a window
        for (Map.Entry<String, VFSListDataset<FImage>> entry : groupedFaces.entrySet()) {
            DisplayUtilities.display(entry.getKey(), entry.getValue());
        }
        FlickrAPIToken flickrToken = DefaultTokenFactory.get(FlickrAPIToken.class);
        // Create a Flickr grey-scale image dataset containing 10 cats
        FlickrImageDataset<FImage> cats = FlickrImageDataset.create(ImageUtilities.FIMAGE_READER,
                flickrToken, "cat", 10);
        // Display the Flickr dataset in a grid-like fashion
        DisplayUtilities.display("Cats", cats);


    }
}
