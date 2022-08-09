package uk.ac.soton.ecs.at2n19.hybridimages;

import Jama.SingularValueDecomposition;
import org.openimaj.image.FImage;
import org.openimaj.image.processing.convolution.FImageConvolveSeparable;
import org.openimaj.image.processor.SinglebandImageProcessor;
import org.openimaj.math.matrix.MatrixUtils;

public class FConvolution implements SinglebandImageProcessor<Float, FImage> {
    public FImage kernel;
    private ConvolveMode mode;

    public FConvolution(FImage kernel) {
        this.kernel = kernel;
        this.setup(false);
    }

    public FConvolution(float[][] kernel) {
        this.kernel = new FImage(kernel);
        this.setup(false);
    }

    public void setBruteForce(boolean brute) {
        this.setup(brute);
    }

    private void setup(boolean brute) {
        if (brute) {
            this.mode = new ConvolveMode.BruteForce(this.kernel);
        } else {
            if (this.kernel.width != 1 && this.kernel.height != 1) {
                SingularValueDecomposition svd = new SingularValueDecomposition(MatrixUtils.matrixFromFloat(this.kernel.pixels));
                if (svd.rank() == 1) {
                    this.mode = new ConvolveMode.Separable(svd);
                } else {
                    this.mode = new ConvolveMode.BruteForce(this.kernel);
                }
            } else {
                this.mode = new ConvolveMode.OneD(this.kernel);
            }

        }
    }

    public void processImage(FImage image) {
        this.mode.convolve(image);
    }

    public float responseAt(int x, int y, FImage image) {
        float sum = 0.0F;
        int kh = this.kernel.height;
        int kw = this.kernel.width;
        int hh = kh / 2;
        int hw = kw / 2;
        int j = 0;

        for(int jj = kh - 1; j < kh; --jj) {
            int i = 0;

            for(int ii = kw - 1; i < kw; --ii) {
                int rx = x + i - hw;
                int ry = y + j - hh;
                sum += image.pixels[ry][rx] * this.kernel.pixels[jj][ii];
                ++i;
            }

            ++j;
        }

        return sum;
    }

    interface ConvolveMode {
        void convolve(FImage var1);

        public static class BruteForce implements ConvolveMode {
            protected FImage kernel;

            BruteForce(FImage kernel) {
                this.kernel = kernel;
            }

            public void convolve(FImage image) {
                int kh = this.kernel.height;
                int kw = this.kernel.width;
                int hh = kh / 2;
                int hw = kw / 2;
                FImage clone = image.newInstance(image.width, image.height);

                for(int y = hh; y < image.height - (kh - hh); ++y) {
                    for(int x = hw; x < image.width - (kw - hw); ++x) {
                        float sum = 0.0F;
                        int j = 0;

                        for(int jj = kh - 1; j < kh; --jj) {
                            int i = 0;

                            for(int ii = kw - 1; i < kw; --ii) {
                                int rx = x + i - hw;
                                int ry = y + j - hh;
                                sum += image.pixels[ry][rx] * this.kernel.pixels[jj][ii];
                                ++i;
                            }

                            ++j;
                        }

                        clone.pixels[y][x] = sum;
                    }
                }

                image.internalAssign(clone);
            }
        }

        public static class Separable implements ConvolveMode {
            private float[] row;
            private float[] col;

            Separable(SingularValueDecomposition svd) {
                int nrows = svd.getU().getRowDimension();
                this.row = new float[nrows];
                this.col = new float[nrows];
                float factor = (float)Math.sqrt(svd.getS().get(0, 0));

                for(int i = 0; i < nrows; ++i) {
                    this.row[i] = (float)svd.getU().get(i, 0) * factor;
                    this.col[i] = (float)svd.getV().get(i, 0) * factor;
                }

            }

            public void convolve(FImage f) {
                FImageConvolveSeparable.convolveHorizontal(f, this.col);
                FImageConvolveSeparable.convolveVertical(f, this.row);
            }
        }

        public static class OneD implements ConvolveMode {
            private float[] kernel;
            private boolean rowMode;

            OneD(FImage image) {
                if (image.height == 1) {
                    this.rowMode = true;
                    this.kernel = image.pixels[0];
                } else {
                    this.rowMode = false;
                    this.kernel = new float[image.height];

                    for(int i = 0; i < image.height; ++i) {
                        this.kernel[i] = image.pixels[i][0];
                    }
                }

            }

            public void convolve(FImage f) {
                if (this.rowMode) {
                    FImageConvolveSeparable.convolveHorizontal(f, this.kernel);
                } else {
                    FImageConvolveSeparable.convolveVertical(f, this.kernel);
                }

            }
        }
    }
}

