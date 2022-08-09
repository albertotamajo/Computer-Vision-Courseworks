package uk.ac.soton.ecs.at2n19.coursework3;

import de.bwaldvogel.liblinear.SolverType;
import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.ListDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.experiment.dataset.sampling.GroupedUniformRandomisedSampler;
import org.openimaj.experiment.dataset.split.GroupedRandomSplitter;
import org.openimaj.experiment.evaluation.classification.ClassificationEvaluator;
import org.openimaj.experiment.evaluation.classification.ClassificationResult;
import org.openimaj.experiment.evaluation.classification.analysers.confusionmatrix.CMAnalyser;
import org.openimaj.experiment.evaluation.classification.analysers.confusionmatrix.CMResult;
import org.openimaj.image.FImage;
import org.openimaj.image.feature.dense.gradient.dsift.DenseSIFT;
import org.openimaj.image.feature.dense.gradient.dsift.PyramidDenseSIFT;
import org.openimaj.io.IOUtils;
import org.openimaj.ml.annotation.ScoredAnnotation;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator;
import org.openimaj.ml.kernel.HomogeneousKernelMap;

import java.io.*;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * Utilities for training a set of linear classifiers on the features extracted by the PHOW extractor.
 */
public class TrainPHOWExtractor {

    /**
     * Produces the annotations for run3 with the parameters that have achieved the highest average accuracy (82.3%)
     * for the computer vision traditional approach. However, due to randomisation, the actual produced annotations
     * could achieve a slightly different accuracy. The produced annotations are not the ones contained in the
     * submitted run3.txt file as those are produced by the deep learning approach.
     * @param args
     * @throws IOException
     */
    public static void main(String[] args) throws IOException {

        GroupedDataset<String, VFSListDataset<FImage>, FImage> trainDataset = Dataset.getTrainDataset();
        DenseSIFT dsift = new DenseSIFT(3, 7);
        PyramidDenseSIFT<FImage> pdsift = new PyramidDenseSIFT<>(dsift, 6f, 4,6,8,10);
        PHOWExtractor<String, ListDataset<FImage>> extractor = new PHOWExtractor<>(484,500, "assigner",
                pdsift, GroupedUniformRandomisedSampler.sample(trainDataset, 1500));

        HomogeneousKernelMap homogeneousKernelMap = new HomogeneousKernelMap(HomogeneousKernelMap.KernelType.Chi2,
                HomogeneousKernelMap.WindowType.Rectangular);
        LiblinearAnnotator<FImage,String> ann = new LiblinearAnnotator<>(
                homogeneousKernelMap.createWrappedExtractor(extractor), LiblinearAnnotator.Mode.MULTICLASS,
                SolverType.L2R_L2LOSS_SVC, 50, 0.00001);
        ann.train(trainDataset);
        classifyTestDataset(ann,"run3.txt");
    }


    /**
     * Annotate the images on the test dataset with the given liblinear annotator. The annotations are saved into a
     * text file whose path is given as argument. The annotations are saved into ascending order based on the name
     * of the image.
     * @param ann liblinear annotator
     * @param path path where to save the annotations
     * @throws IOException
     */
    public static void classifyTestDataset(LiblinearAnnotator<FImage, String> ann, String path) throws IOException {
        List<Pair<Integer,String>> pairs = new ArrayList<>();
        // Create report file
        new File(path).createNewFile();
        VFSListDataset<FImage> testDataset = Dataset.getTestDataset();
        for (int i = 0; i < testDataset.size(); i++){
            FImage image = testDataset.get(i);
            String name = testDataset.getFileObject(i).getName().getBaseName();
            String annotation = classify(ann.annotate(image));
            pairs.add(new Pair<>(Integer.parseInt(name.replace(".jpg","")),annotation));
        }
        pairs = pairs.stream().sorted(Comparator.comparing(x -> x.val1)).collect(Collectors.toList());
        PrintWriter pw = new PrintWriter(new FileWriter(path));
        for (Pair<Integer,String> pair : pairs){
            pw.println(pair.val1 + ".jpg" + " " + pair.val2);
        }
        pw.close();
    }


    /**
     * Return the annotation of the class with the highest confidence from a list of scored annotations
     * @param scoredAnnotations list of scored annotations
     * @return annotation of the class with the highest confidence
     */
    public static String classify(List<ScoredAnnotation<String>> scoredAnnotations){
        float confidence = Float.MIN_VALUE;
        String annotation = null;
        for (ScoredAnnotation<String> scoredAnnotation : scoredAnnotations){
            if (scoredAnnotation.confidence > confidence){
                annotation = scoredAnnotation.annotation;
                confidence = scoredAnnotation.confidence;
            }
        }
        return annotation;
    }

    /**
     * Trains a liblinear classifier with the given extractor on the given training dataset. The liblinear annotator is
     * then saved into the given path.
     * @param extractor PHOW extractor
     * @param trainDataset training dataset
     * @param path path where to save the learnt liblinear annotator
     * @throws IOException
     */
    public static void libLinearPHOW(PHOWExtractor<String, ListDataset<FImage>> extractor,
                                     GroupedDataset<String, VFSListDataset<FImage>, FImage> trainDataset,
                                     String path) throws IOException {

        GroupedRandomSplitter<String, FImage> splits =
                new GroupedRandomSplitter<>(trainDataset, 80, 0, 20);

        HomogeneousKernelMap homogeneousKernelMap = new HomogeneousKernelMap(HomogeneousKernelMap.KernelType.Chi2,
                HomogeneousKernelMap.WindowType.Rectangular);
        System.out.println("here1");
        LiblinearAnnotator<FImage,String> ann = new LiblinearAnnotator<>(
                homogeneousKernelMap.createWrappedExtractor(extractor), LiblinearAnnotator.Mode.MULTICLASS,
                SolverType.L2R_LR, 1, 0.00001);
        ann.train(splits.getTrainingDataset());
        System.out.println("here2");
        // Perform automated evaluation of the linear classifier accuracy and print out a summary report
        ClassificationEvaluator<CMResult<String>, String, FImage> eval = new ClassificationEvaluator<>(
                ann, splits.getTestDataset(), new CMAnalyser<>(CMAnalyser.Strategy.SINGLE));
        Map<FImage, ClassificationResult<String>> guesses = eval.evaluate();
        IOUtils.writeToFile(ann, new File(path));
        CMResult<String> result = eval.analyse(guesses);
        System.out.println(result.getDetailReport());
    }


}