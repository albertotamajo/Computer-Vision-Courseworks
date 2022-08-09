package uk.ac.soton.ecs.at2n19.ch3;

import org.openimaj.image.DisplayUtilities;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.MBFImage;
import org.openimaj.image.colour.ColourSpace;
import org.openimaj.image.connectedcomponent.GreyscaleConnectedComponentLabeler;
import org.openimaj.image.pixel.ConnectedComponent;
import org.openimaj.image.typography.hershey.HersheyFont;
import org.openimaj.ml.clustering.FloatCentroidsResult;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.ml.clustering.kmeans.FloatKMeans;

import java.io.IOException;
import java.net.URL;
import java.util.Arrays;
import java.util.List;

public class Tutorial {
    public static void main(String[] args) throws IOException {

        // Load image from URL
        MBFImage input = ImageUtilities.readMBF(new URL("http://static.openimaj.org/media/tutorial/sinaface.jpg"));
        // Convert the image's current colour space into the LAB colour space
        input = ColourSpace.convert(input, ColourSpace.CIE_Lab);
        // Construct an instance of the KMeans Algorithm which generates 4 different classes
        FloatKMeans cluster = FloatKMeans.createExact(4);
        // Flatten the pixels of the image
        float[][] imageData = input.getPixelVectorNative(new float[input.getWidth() * input.getHeight()][3]);
        // Cluster the image
        FloatCentroidsResult result = cluster.cluster(imageData);
        // Get the centroids of the clusters
        float[][] centroids = result.centroids;
        // Print the LAB coordinates of each centroid
        for (float[] fs : centroids) {
            System.out.println(Arrays.toString(fs));
        }
        // Assign to each pixel the LAB coordinate of its corresponding centroid
        HardAssigner<float[],?,?> assigner = result.defaultHardAssigner();
        for (int y=0; y<input.getHeight(); y++) {
            for (int x=0; x<input.getWidth(); x++) {
                float[] pixel = input.getPixelNative(x, y);
                int centroid = assigner.assign(pixel);
                input.setPixelNative(x, y, centroids[centroid]);
            }
        }

        // Convert the image's colour space back to RGB
        input = ColourSpace.convert(input, ColourSpace.RGB);
        // Create an instance to find the connected components of the image
        GreyscaleConnectedComponentLabeler labeler = new GreyscaleConnectedComponentLabeler();
        // Find the connected components of the image
        List<ConnectedComponent> components = labeler.findComponents(input.flatten());

        // Draw a text at each centroid pixel of every connected component that has an area larger than 600 pixels
        int i = 0;
        for (ConnectedComponent comp : components) {
            if (comp.calculateArea() < 600)
                continue;
            input.drawText("Point:" + (i++), comp.calculateCentroidPixel(), HersheyFont.TIMES_MEDIUM,20);
        }
        // Display the image
        DisplayUtilities.display(input);
    }
}
