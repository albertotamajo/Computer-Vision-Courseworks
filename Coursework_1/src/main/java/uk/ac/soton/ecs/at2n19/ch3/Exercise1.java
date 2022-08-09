package uk.ac.soton.ecs.at2n19.ch3;

import org.openimaj.image.DisplayUtilities;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.MBFImage;
import org.openimaj.image.colour.ColourSpace;
import org.openimaj.image.connectedcomponent.GreyscaleConnectedComponentLabeler;
import org.openimaj.image.pixel.ConnectedComponent;
import org.openimaj.image.processor.PixelProcessor;
import org.openimaj.image.typography.hershey.HersheyFont;
import org.openimaj.ml.clustering.FloatCentroidsResult;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.ml.clustering.kmeans.FloatKMeans;
import org.openimaj.time.Timer;

import java.io.IOException;
import java.net.URL;
import java.util.Arrays;
import java.util.List;

/*
    Rather than looping over the image pixels using two for loops, it is possible to use a PixelProcessor to accomplish
    the same task:

        image.processInplace(new PixelProcessor<Float[]>() {
            Float[] processPixel(Float[] pixel) {
                ...
            }
        });

    Can you re-implement the loop that replaces each pixel with its class centroid using a PixelProcessor?

    What are the advantages and disadvantages of using a PixelProcessor?
*/
public class Exercise1 {
    // The PixelProcessor interface allows to perform operations on each pixel of an image without using two nested
    // for loops to iterate over all the pixels of an image.
    // In the specific case of this exercise, the PixelProcessor interface allows to assign each pixel its corresponding
    // centroid's LAB coordinate without using two nested for loops. This is a clear advantage of the PixelProcessor as
    // it makes the code cleaner and less prone to errors. On the other side, if a person does not know what the
    // PixelProcess does, the two nested for loops are more comprehensible. Furthermore, the two nested for loops seem
    // to be slightly more performant than the PixelProcessor interface. Indeed, the time it took for the PixelProcessor
    // to carry out the task of this exercise is 459ms while the two nested loops required just 203ms. However, based on
    // many repetitions of the task above, the average time difference between the PixelProcessor interface
    // and the two nested loops is slightly smaller.
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
        final float[][] centroids = result.centroids;
        // Print the LAB coordinates of each centroid
        for (float[] fs : centroids) {
            System.out.println(Arrays.toString(fs));
        }


        // Assign to each pixel the LAB coordinate of its corresponding centroid using the PixelProcessor interface
        final HardAssigner<float[],?,?> assigner = result.defaultHardAssigner();
        Timer t1 = Timer.timer();
        input.processInplace(new PixelProcessor<Float[]>() {
            @Override
            public Float[] processPixel(Float[] pixel) {
                int centroid = assigner.assign(tofloatArray(pixel));
                return toFloatArray(centroids[centroid]);
            }
        });
        System.out.println("PixelProcessor : " + t1.duration());

        t1 = Timer.timer();
        for (int y=0; y<input.getHeight(); y++) {
            for (int x=0; x<input.getWidth(); x++) {
                float[] pixel = input.getPixelNative(x, y);
                int centroid = assigner.assign(pixel);
                input.setPixelNative(x, y, centroids[centroid]);
            }
        }
        System.out.println("Nested for loop: " + t1.duration());



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

    // Converts an array of floats into an array of Floats
    public static Float[] toFloatArray(float[] array){
        Float[] array2 = new Float[array.length];
        for (int i = 0; i < array.length; i++){
            array2[i] = array[i];
        }
        return array2;
    }

    // Converts an array of Floats into an array of floats
    public static float[] tofloatArray(Float[] array){
        float[] array2 = new float[array.length];
        for (int i = 0; i < array.length; i++){
            array2[i] = array[i];
        }
        return array2;
    }

}
