package uk.ac.soton.ecs.at2n19.ch6;

import org.apache.commons.vfs2.FileSystemException;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.image.DisplayUtilities;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/*
    Using the faces dataset available from http://datasets.openimaj.org/att_faces.zip, can you display an image that
    shows a randomly selected photo of each person in the dataset?
 */
public class Exerice1 {
    public static void main(String[] args) throws FileSystemException {

        // Create a group dataset backed by a zip file hosted in a web-server
        VFSGroupDataset<FImage> groupedFaces = new VFSGroupDataset<FImage>(
                "zip:http://datasets.openimaj.org/att_faces.zip", ImageUtilities.FIMAGE_READER);
        List<FImage> randomFacesForEachPerson = new ArrayList<>();
        // Append a randomly selected image of each person to a list
        for (Map.Entry<String, VFSListDataset<FImage>> entry : groupedFaces.entrySet()) {
            randomFacesForEachPerson.add(entry.getValue().getRandomInstance());
        }
        // Display a randomly selected image of each person
        DisplayUtilities.display("Random faces",randomFacesForEachPerson);
    }
}
