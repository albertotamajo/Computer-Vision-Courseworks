package uk.ac.soton.ecs.at2n19.hybridimages;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

public class Convolution {

    static{System.loadLibrary(Core.NATIVE_LIBRARY_NAME);}
    public static float[][] convolve(float[][] image, float[][] kernel){
        Mat imageMatrix = toMat(image);
        Mat kernelMatrix = toMat(kernel);
        Core.flip(kernelMatrix, kernelMatrix, -1);
        Mat dest = new Mat();
        Imgproc.filter2D(imageMatrix,dest,-1,kernelMatrix,new Point(-1,-1), 0,0);
        return toArray(dest);

    }

    public static Mat toMat(float[][] array){
        Mat mat = new Mat(array.length, array[0].length, CvType.CV_32F);
        for(int i=0; i < array.length; i++){
            for(int c=0; c < array[0].length; c++){
                mat.put(i,c,array[i][c]);
            }
        }
        return mat;
    }

    public static float[][] toArray(Mat mat){
        float[] flat = new float[mat.height()*mat.width()];
        mat.get(0,0,flat);
        float[][] output = new float[mat.height()][mat.width()];
        int incr = 0;
        for(int i=0; i < output.length; i++){
            for(int c=0; c < output[0].length; c++){
                output[i][c] = flat[incr];
                incr++;
            }
        }
        return output;
    }
}
