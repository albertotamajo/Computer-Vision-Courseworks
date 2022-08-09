package uk.ac.soton.ecs.at2n19.ch1;

import org.openimaj.image.DisplayUtilities;
import org.openimaj.image.MBFImage;
import org.openimaj.image.colour.ColourSpace;
import org.openimaj.image.colour.RGBColour;
import org.openimaj.image.processing.convolution.FGaussianConvolve;
import org.openimaj.image.typography.hershey.HersheyFont;

/**
 Take a look at the App.java from within your IDE. Can you modify the code to render something other than “hello world”
 in a different font and colour?
 */
public class Exercise1 {
    public static void main( String[] args ) {
    	//Create an image
        // The size of the image is increased so that to fit the new text "Hello Computer Vision"
        MBFImage image = new MBFImage(800,100, ColourSpace.RGB);

        //Fill the image with white
        image.fill(RGBColour.WHITE);
        		        
        //Render some text into the image
        // The new text is "Hello Computer Vision"
        // The new font is TIMES MEDIUM ITALIC
        // The new color is GREEN
        image.drawText("Hello Computer Vision", 10, 60, HersheyFont.TIMES_MEDIUM_ITALIC, 50, RGBColour.GREEN);

        //Apply a Gaussian blur
        image.processInplace(new FGaussianConvolve(2f));
        
        //Display the image
        DisplayUtilities.display(image);
    }
}
