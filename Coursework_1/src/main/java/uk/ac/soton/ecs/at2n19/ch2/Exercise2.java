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
    Those speech bubbles look rather plain; why not give them a nice border?
*/
public class Exercise2 {
    public static void main(String[] args) throws IOException {

        // Read a multi-band image from a URL
        MBFImage image = ImageUtilities.readMBF(new URL("http://static.openimaj.org/media/tutorial/sinaface.jpg"));
        // Detect the image's edges using the Canny Edge Detector algorithm
        image.processInplace(new CannyEdgeDetector());

        // Draw an ellipse filled with white
        image.drawShapeFilled(new Ellipse(700f, 450f, 20f, 10f, 0f), RGBColour.WHITE);
        // Draw a magenta border around the above ellipse
        image.drawPolygon(new Ellipse(700f, 450f, 20f, 10f, 0f).asPolygon(),2, RGBColour.MAGENTA);

        // Draw an ellipse filled with white
        image.drawShapeFilled(new Ellipse(650f, 425f, 25f, 12f, 0f), RGBColour.WHITE);
        // Draw a magenta border around the above ellipse
        image.drawPolygon(new Ellipse(650f, 425f, 25f, 12f, 0f).asPolygon(),2, RGBColour.MAGENTA);

        // Draw an ellipse filled with white
        image.drawShapeFilled(new Ellipse(600f, 380f, 30f, 15f, 0f), RGBColour.WHITE);
        // Draw a magenta border around the above ellipse
        image.drawPolygon(new Ellipse(600f, 380f, 30f, 15f, 0f).asPolygon(),2, RGBColour.MAGENTA);

        // Draw an ellipse filled with white
        image.drawShapeFilled(new Ellipse(500f, 300f, 100f, 70f, 0f), RGBColour.WHITE);
        // Draw a magenta border around the above ellipse
        image.drawPolygon(new Ellipse(500f, 300f, 100f, 70f, 0f).asPolygon(),2, RGBColour.MAGENTA);

        // Draw text inside the biggest ellipse
        image.drawText("OpenIMAJ is", 425, 300, HersheyFont.ASTROLOGY, 20, RGBColour.BLACK);
        // Draw text inside the biggest ellipse
        image.drawText("Awesome", 425, 330, HersheyFont.ASTROLOGY, 20, RGBColour.BLACK);

        // Display the image
        DisplayUtilities.display(image);
    }
}
