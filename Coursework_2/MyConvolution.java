package uk.ac.soton.ecs.at2n19.hybridimages;

import org.openimaj.image.FImage;
import org.openimaj.image.processor.SinglebandImageProcessor;


public class MyConvolution implements SinglebandImageProcessor<Float, FImage> {
    private float[][] kernel;
    private int k_rows;
    private int k_cols;

    public MyConvolution(float[][] kernel) {
        this.kernel = kernel;
        this.k_rows = kernel.length;  // number of rows in the kernel
        this.k_cols = kernel[0].length;  // number of columns in the kernel
    }

    @Override
    public void processImage(FImage image) {
        // convolve image with kernel and store result back in image
        int image_rows = image.getHeight();  // number of rows in the image
        int image_cols = image.getWidth();  // number of cols in the image

        // if the number of kernel columns is 1 then the padding width must be 0
        // otherwise it must be equal to half of (the number of columns - 1)
        int paddingWidth = k_cols == 1 ? 0 : (k_cols - 1)/2;

        // if the number of kernel rows is 1 then the padding height must be 0
        // otherwise it must be equal to half of (the number of rows - 1)
        int paddingHeight = k_rows == 1 ? 0 : (k_rows - 1)/2;

        FImage paddedImage = image.padding(paddingWidth, paddingHeight,0.0f);
        float [][] pixels = paddedImage.pixels;

        float[][] convolvedPixels = convolve(pixels, image_rows, image_cols);
        image.internalAssign(new FImage(convolvedPixels));

    }

    // pixels is a float array containing the values of the pixels of a padded image
    // rows is the number of rows in the returned array which represents the height of the image resulting from the convolution
    // cols is the number of columns in the returned array which represents the width of the image resulting from the convolution
    private float[][] convolve(float[][] paddedImage, int rows, int cols){
        float[][] output = new float[rows][cols];  // output of the method
        int outputRowIndex = 0;
        int outputColIndex = 0;
        int paddedImageRows = paddedImage.length;  // number of rows in the padded image
        int paddedImageCols = paddedImage[0].length;  // number of cols in the padded image
        for (int r = 0; r + k_rows <= paddedImageRows ; r++){
            for(int c = 0; c + k_cols <= paddedImageCols; c++){
                output[outputRowIndex][outputColIndex] = kernelWeightedSum(paddedImage, r, c);
                if((outputColIndex + 1) < cols ){
                    outputColIndex++;
                }else{
                    outputColIndex = 0;
                    outputRowIndex++;
                }
            }
        }
        return output;
    }

    // r and c are the coordinates of the upperleft pixel that needs to be convolved by the kernel
    private float kernelWeightedSum(float[][] paddedImage, int r, int c){
        int endRow = r + k_rows;
        int endCol = c + k_cols;
        int kernelRowIndex = k_rows - 1;
        int kernelColIndex = k_cols - 1;
        float weightedSum = 0.0f;
        for(int row = r; row < endRow; row++){
            for(int col = c; col < endCol; col++){
                weightedSum += paddedImage[row][col] * kernel[kernelRowIndex][kernelColIndex];
                if((kernelColIndex - 1) >= 0){
                    kernelColIndex--;
                }else{
                    kernelColIndex = k_cols - 1;
                    kernelRowIndex--;
                }

            }
        }
        return weightedSum;
    }
}
