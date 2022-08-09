package uk.ac.soton.ecs.at2n19.coursework3;

import org.openimaj.data.DataSource;
import org.openimaj.data.dataset.Dataset;
import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.ListDataset;
import org.openimaj.experiment.dataset.sampling.GroupedUniformRandomisedSampler;
import org.openimaj.feature.DoubleFV;
import org.openimaj.feature.FeatureExtractor;
import org.openimaj.feature.SparseIntFV;
import org.openimaj.feature.local.data.LocalFeatureListDataSource;
import org.openimaj.feature.local.list.LocalFeatureList;
import org.openimaj.image.FImage;
import org.openimaj.image.feature.dense.gradient.dsift.ByteDSIFTKeypoint;
import org.openimaj.image.feature.dense.gradient.dsift.PyramidDenseSIFT;
import org.openimaj.image.feature.local.aggregate.BagOfVisualWords;
import org.openimaj.image.feature.local.aggregate.PyramidSpatialAggregator;
import org.openimaj.image.processing.convolution.FSobel;
import org.openimaj.io.IOUtils;
import org.openimaj.ml.clustering.ByteCentroidsResult;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.ml.clustering.kmeans.ByteKMeans;
import org.openimaj.util.pair.IntFloatPair;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

/**
 * Extract Pyramid Histogram of Words from an image based on a codebook learnt with a given training dataset.
 * The codebook is learnt by clustering features extracted from a set of training dataset samples.
 * The features are extracted from the training dataset samples using the PHOW technique.
 * @param <T> KEY
 * @param <Q> List dataset
 */
public class PHOWExtractor<T,Q extends ListDataset<FImage>> implements FeatureExtractor<DoubleFV, FImage> {

    private final PyramidDenseSIFT<FImage> pdsift;
    private final PyramidSpatialAggregator<byte[], SparseIntFV> spatial;

    /**
     * Construct the extractor so that to learn a codebook containing the given number of visual words.
     * The codebook is learnt from a set of samples drawn from the given training dataset.
     * The features are extracted from the training dataset samples using the PHOW technique. This is why
     * a pyramid Dense SIFT extractor needs to be provided as well. If the provided path refers to an assigner,
     * then that file is loaded, otherwise after learning the codebook, the assigner is saved into that path.
     * @param visualWords number of visual words
     * @param samples number of samples
     * @param pathAssigner path where to save/load the assigner
     * @param pdsift pyramid Dense sift extractor
     * @param dataset training dataset
     * @throws IOException
     */
    public PHOWExtractor(int visualWords, int samples, String pathAssigner, PyramidDenseSIFT<FImage> pdsift,
                         GroupedDataset<T,Q,FImage> dataset) throws IOException {
        this.pdsift = pdsift;
        if (Files.exists(Paths.get(pathAssigner))){
            HardAssigner<byte[], float[], IntFloatPair> assigner = IOUtils.readFromFile(new File(pathAssigner));
            BagOfVisualWords<byte[]> bovw = new BagOfVisualWords<>(assigner);
            this.spatial = new PyramidSpatialAggregator<>(bovw,2,4);
        }else {
            this.spatial = trainQuantiser(visualWords, pathAssigner,
                    GroupedUniformRandomisedSampler.sample(dataset, samples));
        }
    }

    /**
     * Learn the codebook of visual words for a given dataset.
     * The features are extracted from the given dataset samples using the PHOW technique.
     * The learnt assigner is saved into the given path.
     * @param visualWords number of visual words
     * @param pathAssigner path where to save the assigner
     * @param dataset training dataset
     * @return assigner
     * @throws IOException
     */
    private PyramidSpatialAggregator<byte[], SparseIntFV> trainQuantiser(int visualWords, String pathAssigner,
                                                                         Dataset<FImage> dataset) throws IOException {

        List<LocalFeatureList<ByteDSIFTKeypoint>> allkeys = new ArrayList<>();

        for (FImage image : dataset) {
            pdsift.analyseImage(image);
            allkeys.add(pdsift.getByteKeypoints(0.005f));
        }


        ByteKMeans km = ByteKMeans.createKDTreeEnsemble(visualWords);
        DataSource<byte[]> datasource = new LocalFeatureListDataSource<>(allkeys);
        ByteCentroidsResult result = km.cluster(datasource);
        HardAssigner<byte[], float[], IntFloatPair> assigner = result.defaultHardAssigner();
        IOUtils.writeToFile(assigner, new File(pathAssigner));
        BagOfVisualWords<byte[]> bovw = new BagOfVisualWords<>(assigner);
        return new PyramidSpatialAggregator<>(bovw,2,4);
    }

    /**
     * Extract features from the given image
     * @param image image to extract features from
     * @return Pyramid Histogram of Words vector
     */
    @Override
    public DoubleFV extractFeature(FImage image) {
        pdsift.analyseImage(image);
        return spatial.aggregate(pdsift.getByteKeypoints(0.015f), image.getBounds()).normaliseFV();
    }

    public static void main(String[] args) {
        FSobel sobel = new FSobel();
        sobel.analyseImage(new FImage(new float[]{0,0,0,0,1,0,0,0,1,1,0,0,1,1,1,0,0,0,1,1,0,0,0,0,1},5,5));
        System.out.println(sobel.dy);
    }

}