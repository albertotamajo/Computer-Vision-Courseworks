package uk.ac.soton.ecs.at2n19.hybridimages;

import org.openimaj.image.FImage;
import org.openimaj.image.processor.SinglebandImageProcessor;

public class ConvolutionGiovanni implements SinglebandImageProcessor<Float, FImage> {
    private final float[][] kernel;

    public ConvolutionGiovanni(float[][] kernel) {
        this.kernel = kernel;
    }


    /**
     * This method applies convolution the image with kernel and store result back in image.
     *
     * @param image         The image to apply convolution to
     */
    @Override
    public void processImage(FImage image) {

        // Zero-padding the image with required pixels to allow the convolution to reach the edges
        int padWidth = (kernel[0].length - 1) / 2;
        int padHeight = (kernel.length - 1) / 2;
        FImage temp = image.padding(padWidth, padHeight, 0.0f);

        // Flipping Kernel
        float[][] flippedKernel = flipKernel(kernel);

        // Initialising convolution result
        FImage convolutedImage = new FImage(new float[image.getRows()][image.getCols()]);

        // Performing convolution
        for (int row = 0; row < convolutedImage.height; row++) {
            for (int col = 0; col < convolutedImage.width; col++) {
                float acc = 0F;
                for (int r = 0; r < flippedKernel.length; r++) {
                    for (int c=0; c< flippedKernel[0].length; c++) {
                        acc += temp.getPixel(col+c, row+r) * flippedKernel[r][c];
                    }
                }
                convolutedImage.setPixel(col, row, acc);
            }
        }

        // Setting the contents of the temporary buffer image to the real image
        image.internalAssign(convolutedImage);
    }



    /**
     * This method, given a kernel, returns its flipped version
     *
     * @param kernel        The kernel to flip
     * @return              The flipped kernel
     */
    public float[][] flipKernel(float[][] kernel) {
        float[][] flippedKernel = new float[kernel.length][kernel[0].length];
        for (int row = 0; row < kernel.length; row++) {
            for (int col = 0; col < kernel[0].length; col++) {
                flippedKernel[row][col] = kernel[kernel.length-row-1][kernel[0].length-col-1];
            }
        }
        return flippedKernel;
    }
}

