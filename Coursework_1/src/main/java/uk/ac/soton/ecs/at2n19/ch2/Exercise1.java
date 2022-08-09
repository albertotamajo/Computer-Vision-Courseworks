package uk.ac.soton.ecs.at2n19.ch2;

import org.openimaj.image.DisplayUtilities;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.MBFImage;
import org.openimaj.image.colour.RGBColour;
import org.openimaj.image.processing.edges.CannyEdgeDetector;
import org.openimaj.image.typography.hershey.HersheyFont;
import org.openimaj.math.geometry.shape.Ellipse;

import java.io.IOException;
import java.net.URL;

/*
    Opening lots of windows can waste time and space (for example if you wanted to view images on every iteration of a
    process within a loop). In OpenIMAJ we provide a facility to open a named display so that was can reuse the display
    referring to it by name. Try to do this with all the images we display in this tutorial. Only 1 window should open
    for the whole tutorial.
*/
public class Exercise1 {
    public static void main(String[] args) throws IOException {

        // Read a multi-band image from a URL
        MBFImage image = ImageUtilities.readMBF(new URL("http://static.openimaj.org/media/tutorial/sinaface.jpg"));
        // Display the image in a frame called "main"
        DisplayUtilities.displayName(image,"main");
        // Display the image's first band in a frame called "main" (the image's red channel in this case since the image's colour space is RGB)
        DisplayUtilities.displayName(image.getBand(0), "main");
        // Display the image's second band in a frame called "main" (the image's green channel in this case since the image's colour space is RGB)
        DisplayUtilities.displayName(image.getBand(1), "main");
        // Display the image's third band in a frame called "main" (the image's blue channel in this case since the image's colour space is RGB)
        DisplayUtilities.displayName(image.getBand(2), "main");

        // Clone the image
        MBFImage clone = image.clone();
        // Set all clone's second band pixels to 0
        clone.getBand(1).fill(0f);
        // Set all clone's third band pixels to 0
        clone.getBand(2).fill(0f);
        // Display the clone image in a frame called "main"
        DisplayUtilities.displayName(clone, "main");

        // Set all clone's second and third band pixels to 0
        for (int y=0; y<image.getHeight(); y++) {
            for(int x=0; x<image.getWidth(); x++) {
                clone.getBand(1).pixels[y][x] = 0;
                clone.getBand(2).pixels[y][x] = 0;
            }
        }
        // Display the clone in a frame called "main"
        DisplayUtilities.displayName(clone,"main");

        // Detect the image's edges using the Canny Edge Detector algorithm
        image.processInplace(new CannyEdgeDetector());
        // Display the image in a frame called "main"
        DisplayUtilities.displayName(image, "main");

        // Draw an ellipse filled with white
        image.drawShapeFilled(new Ellipse(700f, 450f, 20f, 10f, 0f), RGBColour.WHITE);
        // Draw an ellipse filled with white
        image.drawShapeFilled(new Ellipse(650f, 425f, 25f, 12f, 0f), RGBColour.WHITE);
        // Draw an ellipse filled with white
        image.drawShapeFilled(new Ellipse(600f, 380f, 30f, 15f, 0f), RGBColour.WHITE);
        // Draw an ellipse filled with white
        image.drawShapeFilled(new Ellipse(500f, 300f, 100f, 70f, 0f), RGBColour.WHITE);
        // Draw text inside the biggest ellipse
        image.drawText("OpenIMAJ is", 425, 300, HersheyFont.ASTROLOGY, 20, RGBColour.BLACK);
        // Draw text inside the biggest ellipse
        image.drawText("Awesome", 425, 330, HersheyFont.ASTROLOGY, 20, RGBColour.BLACK);
        // Display the image in a frame called "main"
        DisplayUtilities.displayName(image,"main");
    }
}
