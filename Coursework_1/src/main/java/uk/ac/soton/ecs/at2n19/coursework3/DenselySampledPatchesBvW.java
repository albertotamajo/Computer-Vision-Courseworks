package uk.ac.soton.ecs.at2n19.coursework3;

import org.openimaj.data.DataSource;
import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.ListDataset;
import org.openimaj.experiment.dataset.split.GroupedRandomSplitter;
import org.openimaj.feature.DoubleFV;
import org.openimaj.feature.FeatureExtractor;
import org.openimaj.feature.local.data.LocalFeatureListDataSource;
import org.openimaj.feature.local.list.LocalFeatureList;
import org.openimaj.image.FImage;
import org.openimaj.image.feature.FImage2DoubleFV;
import org.openimaj.image.feature.local.aggregate.BagOfVisualWords;
import org.openimaj.image.processing.algorithm.MeanCenter;
import org.openimaj.io.IOUtils;
import org.openimaj.ml.clustering.DoubleCentroidsResult;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.ml.clustering.kmeans.DoubleKMeans;
import org.openimaj.data.dataset.Dataset;
import org.openimaj.util.pair.IntDoublePair;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;


/**
 * Extract bag-of-visual-words feature from an image based on a codebook learnt with a given training dataset.
 * The codebook is learnt by clustering features extracted from a set of training dataset samples.
 * The features are extracted from the training dataset samples using a sliding window feature extractor which mean
 * centers, normalises and then flattens the regions encountered.
 * @param <T> KEY
 * @param <Q> List dataset
 */
public class DenselySampledPatchesBvW<T,Q extends ListDataset<FImage>> implements FeatureExtractor<DoubleFV, FImage> {

    private final SlidingWindowExtractor<DoubleFV> slidingWindowExtractor;
    private final HardAssigner<double[], double[], IntDoublePair> assigner;


    /**
     * Construct the extractor so that to learn a codebook containing the given number of visual words.
     * The codebook is learnt from a set of samples drawn from the given training dataset.
     * The features are extracted from the training dataset samples using a sliding window of the given width and
     * height. The window slides along the x and y directions according to the given steps.
     * @param w width of the window
     * @param h height of the window
     * @param stepX step along the x direction
     * @param stepY step along the y direction
     * @param visualWords number of visual words
     * @param samples number of samples drawn from the training dataset to be used for the creation of the codebook
     * @param dataset training dataset
     */
    public DenselySampledPatchesBvW(int w, int h, int stepX, int stepY, int visualWords, int samples,
                                    GroupedDataset<T,Q,FImage> dataset) {

        this.slidingWindowExtractor = new SlidingWindowExtractor<>(w,h,stepX,stepY, window ->
                new FImage2DoubleFV().extractFeature(window.process(new MeanCenter())));

        GroupedRandomSplitter<T,FImage> splits =
                new GroupedRandomSplitter<>(dataset, samples/15, 0, 20);
        this.assigner = trainQuantiser(visualWords, splits.getTrainingDataset());
    }


    /**
     * Construct the extractor so that to learn a codebook containing the given number of visual words.
     * The codebook is learnt from a set of samples drawn from the given training dataset.
     * The features are extracted from the training dataset samples using a sliding window of the given width and
     * height. The window slides along the x and y directions according to the given steps. The Bag of Visual Words
     * extractor is saved in the given path.
     * @param w width of the window
     * @param h height of the window
     * @param stepX step along the x direction
     * @param stepY step along the y direction
     * @param visualWords number of visual words
     * @param samples number of samples drawn from the training dataset to be used for the creation of the codebook
     * @param bovwPath path where to save the hard assigner
     * @param dataset training dataset
     * @throws IOException
     */
    public DenselySampledPatchesBvW(int w, int h, int stepX, int stepY, int visualWords, int samples, String bovwPath,
                                    GroupedDataset<T,Q,FImage> dataset) throws IOException {

        this(w,h,stepX,stepY,visualWords,samples,dataset);
        IOUtils.writeToFile(assigner,new File(bovwPath));
    }


    /**
     * Construct the extractor by loading a previously saved Bag of Visual Words extractor.
     * The features are extracted from the training dataset samples using a sliding window of the given width and
     * height. The window slides along the x and y directions according to the given steps.
     * @param w width of the window
     * @param h height of the window
     * @param stepX step along the x direction
     * @param stepY step along the y direction
     * @param bovwPath path where to load the hard assigner
     * @throws IOException
     */
    public DenselySampledPatchesBvW(int w, int h, int stepX, int stepY, String bovwPath) throws IOException {

        this.slidingWindowExtractor = new SlidingWindowExtractor<>(w,h,stepX,stepY, window ->
                new FImage2DoubleFV().extractFeature(window.process(new MeanCenter())));

        this.assigner = IOUtils.readFromFile(new File(bovwPath));
    }


    /**
     * Learn the codebook of visual words for a given dataset.
     * The features are extracted from the given dataset samples using a sliding window feature extractor which mean
     * centers, normalises and then flattens the regions encountered.
     * @param visualWords number of visual words
     * @param dataset dataset to be used for learning the codebook of visual words
     * @return instance that extracts a bag-of-visual-words vector for a given image
     */
    private HardAssigner<double[], double[], IntDoublePair> trainQuantiser(int visualWords, Dataset<FImage> dataset) {

        List<LocalFeatureList<SlidingWindowLocalFeature<DoubleFV>>> allLocalFeats = new ArrayList<>();

        // Extract features from each image in the dataset and add them to the list of local features
        for (FImage image : dataset) {
            allLocalFeats.add(slidingWindowExtractor.extractFeature(image));
        }

        //if (allLocalFeats.size() > 10000)
        //    allLocalFeats = allLocalFeats.subList(0, 10000);

        // Cluster the extracted local features using KMeans, use the resulting centroids to construct an assigner
        // to be used by a BagOfVisualWords instance. This instance creates a bag-of-visual-words vector for a given
        // image.
        DoubleKMeans km = DoubleKMeans.createKDTreeEnsemble(visualWords);
        DataSource<double[]> datasource = new LocalFeatureListDataSource<>(allLocalFeats);
        DoubleCentroidsResult result = km.cluster(datasource);
        return result.defaultHardAssigner();
    }


    /**
     * Extract features from the given image
     * @param image image to extract features from
     * @return normalised bag-of-visual-words vector for the given image
     */
    @Override
    public DoubleFV extractFeature(FImage image) {
        BagOfVisualWords<double[]> bovw = new BagOfVisualWords<>(assigner);
        // Return the normalised bag-of-visual-words vector for the given image
        return bovw.aggregate(slidingWindowExtractor.extractFeature(image)).normaliseFV(2);
    }
}
