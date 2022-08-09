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

public class Tutorial {
    public static void main( String[] args ) throws IOException {

        // Read a multi-band image from a URL
        MBFImage image = ImageUtilities.readMBF(new URL("http://static.openimaj.org/media/tutorial/sinaface.jpg"));
        // Print the image's colour space
        System.out.println(image.colourSpace);
        // Display the image
        DisplayUtilities.display(image);

        // Display the image's first band (the image's red channel in this case since the image's colour space is RGB)
        DisplayUtilities.display(image.getBand(0), "Red Channel");
        // Display the image's second band (the image's green channel in this case since the image's colour space is RGB)
        DisplayUtilities.display(image.getBand(1), "Green Channel");
        // Display the image's third band (the image's blue channel in this case since the image's colour space is RGB)
        DisplayUtilities.display(image.getBand(2), "Blue Channel");

        // Clone the image
        MBFImage clone = image.clone();
        // Set all clone's second band pixels to 0
        clone.getBand(1).fill(0f);
        // Set all clone's third band pixels to 0
        clone.getBand(2).fill(0f);
        // Display the clone image
        DisplayUtilities.display(clone);

        // Set all clone's second and third band pixels to 0
        for (int y=0; y<image.getHeight(); y++) {
            for(int x=0; x<image.getWidth(); x++) {
                clone.getBand(1).pixels[y][x] = 0;
                clone.getBand(2).pixels[y][x] = 0;
            }
        }
        // Display the clone image
        DisplayUtilities.display(clone);

        // Detect the image's edges using the Canny Edge Detector algorithm
        image.processInplace(new CannyEdgeDetector());
        // Display the image
        DisplayUtilities.display(image);

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
        // Display the image
        DisplayUtilities.display(image);
    }
}
