package uk.ac.soton.ecs.at2n19.hybridimages;

import org.openimaj.image.FImage;
import org.openimaj.image.processor.SinglebandImageProcessor;

public class ConvolutionAlessandro implements SinglebandImageProcessor<Float, FImage> {
    private float[][] kernel;
    private final float[][] flippedKernel;
    private final int x_padding;
    private final int y_padding;

    public ConvolutionAlessandro(float[][] kernel) {
        this.kernel = kernel;
        // Calculates padding for image
        this.x_padding = (kernel[0].length - 1) / 2;
        this.y_padding = (kernel.length - 1) / 2;
        // Flips Kernel
        float[][] flippedKernel = new float[kernel.length][kernel[0].length];
        for(int row=0; row<kernel.length; row++){
            for(int col=0; col<kernel[0].length; col++){
                flippedKernel[row][col] = kernel[kernel.length-row-1][kernel[0].length-col-1];
            }
        }
        this.flippedKernel = flippedKernel;
    }

    @Override
    public void processImage(FImage image) {
        FImage paddedImage = image.padding(x_padding,y_padding,0f);
        FImage convolutionResult = new FImage(new float[image.getRows()][image.getCols()]);

        for(int row=0; row<convolutionResult.height; row++){
            for(int col=0; col<convolutionResult.width; col++){
                float accumulator = 0f;
                for(int krow=0; krow<this.flippedKernel.length; krow++){
                    for(int kcol=0; kcol<this.flippedKernel[0].length; kcol++){
                        accumulator += paddedImage.getPixel(col+kcol, row+krow) * flippedKernel[krow][kcol];
                    }
                }
                convolutionResult.setPixel(col, row, (float) accumulator);
            }
        }
        image.internalAssign(convolutionResult);
    }
}
