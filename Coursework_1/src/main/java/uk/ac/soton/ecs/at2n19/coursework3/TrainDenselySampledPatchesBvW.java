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
import org.openimaj.experiment.validation.cross.CrossValidationIterable;
import org.openimaj.experiment.validation.cross.GroupedKFold;
import org.openimaj.image.FImage;
import org.openimaj.ml.annotation.ScoredAnnotation;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * Utilities for training a set of linear classifiers on the features extracted by the DenselySampledPatchesBvW
 * extractor.
 */
public class TrainDenselySampledPatchesBvW {


    /**
     * Produces the run2.txt file with the parameters that have achieved the highest average accuracy (66.3%)
     * on the 5-fold cross-validation iterations. This code should replicate the annotations produced in the
     * submitted run2.txt file. However, due to randomisation, the actual produced annotations could be slightly
     * different than the ones submitted.
     * @param args
     * @throws IOException
     */
    public static void main(String[] args) throws IOException {
        GroupedDataset<String, VFSListDataset<FImage>, FImage> trainDataset = Dataset.getTrainDataset();
        DenselySampledPatchesBvW<String,ListDataset<FImage>> extractor = new DenselySampledPatchesBvW<>(3,3,
                4,4,700,750,GroupedUniformRandomisedSampler.sample(trainDataset,1500));
        LiblinearAnnotator<FImage, String> ann = new LiblinearAnnotator<>(
                extractor, LiblinearAnnotator.Mode.MULTICLASS, SolverType.L2R_L2LOSS_SVC, 1.0, 0.00001);
        ann.train(trainDataset);
        classifyTestDataset(ann,"run2.txt");
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
     * Perform K-fold cross validation on a given dataset for several set of linear classifiers that use different
     * parameters for the feature extraction process.
     * Features from the train dataset are extracted by the DenselySampledPatchesBvW extractor.
     * The performance of each model is appended to a report.
     * @param minw minimum width of the window
     * @param maxw maximum width of the window
     * @param stepW incrementing step for the window width
     * @param minStepX minimum step along the x direction (also along the y direction)
     * @param maxStepX maximum step along the x direction (also along the y direction)
     * @param stepStepX incrementing step for the step along the x direction (also along the y direction)
     * @param minVisualWords minimum number of visual words
     * @param maxVisualWords maximum number of visual words
     * @param stepVisualWords incrementing step for the number of visual words
     * @param minNsamples minimum number of samples drawn from the training dataset to be used for the creation of the
     *                    codebook
     * @param maxNsamples maximum number of samples drawn from the training dataset to be used for the creation of the
     *                    codebook
     * @param stepSamples incrementing step for the number of samples
     * @param nfolds number of folds
     * @param dataset dataset
     * @param filePath path to the report file
     * @throws IOException
     */
    public static void evaluateModelsCrossValidation(int minw, int maxw, int stepW, int minStepX, int maxStepX,
                                                     int stepStepX, int minVisualWords, int maxVisualWords,
                                                     int stepVisualWords, int minNsamples, int maxNsamples,
                                                     int stepSamples,  int nfolds, GroupedDataset<String,
                                                     ListDataset<FImage>, FImage> dataset,
                                                     String filePath) throws IOException {
        // Create report file
        new File(filePath).createNewFile();
        PrintWriter pw = new PrintWriter(new FileWriter(filePath));
        // Print the header on the report file
        pw.println("W,StepX,VisualWords,Samples,Accuracy,Step W,Step StepX,Step VisualWords,Step Samples,Folds");
        pw.close();
        for(int w = minw; w <= maxw; w+=stepW) {
            for (int stepX = minStepX; stepX <= maxStepX; stepX += stepStepX) {
                for (int visualWords = minVisualWords; visualWords <= maxVisualWords; visualWords += stepVisualWords) {
                    for (int nsamples = minNsamples; nsamples <= maxNsamples; nsamples += stepSamples) {
                        // Perform cross validation on the given model
                        double avgAccuracy = crossValidation(w, w, stepX, stepX, visualWords, nsamples, nfolds, dataset);
                        pw = new PrintWriter(new FileWriter(filePath, true));
                        // Append to the report file the performance of the given model
                        pw.println(w + "," + stepX + "," + visualWords + "," + nsamples + "," + avgAccuracy + "," +
                                stepW + "," + stepStepX + "," + stepVisualWords + "," + stepSamples + "," + nfolds);
                        pw.close();
                    }
                }
            }
        }
    }


    /**
     * Train a set of linear classifiers on the given train dataset.
     * Features from the train dataset are extracted by the DenselySampledPatchesBvW extractor according to the
     * provided parameters.
     * The performance of the trained model is evaluated on the given validation dataset.
     * @param w width of the window
     * @param h height of the window
     * @param stepX step along the x direction
     * @param stepY step along the y direction
     * @param visualWords number of visual words
     * @param samples number of samples drawn from the training dataset to be used for the creation of the codebook
     * @param trainDataset train dataset
     * @param valDataset validation dataset
     * @return accuracy on the validation dataset
     */
    public static double evaluateModel(int w, int h, int stepX, int stepY, int visualWords, int samples,
                                       GroupedDataset<String, ListDataset<FImage>, FImage> trainDataset,
                                       GroupedDataset<String, ListDataset<FImage>, FImage> valDataset){

        DenselySampledPatchesBvW<String, ListDataset<FImage>> extractor = new DenselySampledPatchesBvW<>(w, h, stepX,
                stepY, visualWords, samples, trainDataset);
        // Create a set of linear classifiers
        LiblinearAnnotator<FImage, String> ann = new LiblinearAnnotator<>(
                extractor, LiblinearAnnotator.Mode.MULTICLASS, SolverType.L2R_L2LOSS_SVC, 1.0, 0.00001);
        // Train the set of linear classifiers
        ann.train(trainDataset);

        ClassificationEvaluator<CMResult<String>, String, FImage> eval = new ClassificationEvaluator<>(ann,
                valDataset, new CMAnalyser<>(CMAnalyser.Strategy.SINGLE));
        // Evaluate the performance of the trained model
        Map<FImage, ClassificationResult<String>> guesses = eval.evaluate();
        CMResult<String> result = eval.analyse(guesses);
        System.out.println(result.getDetailReport());
        // Return the accuracy achieved on the validation dataset
        return result.getMatrix().getAccuracy();
    }

    /**
     * Train a set of linear classifiers on the training dataset for all possible combinations of the given patch
     * sizes, vocabulary sizes and direction steps. For each combination, a DenselySampledPatchesBvW class is
     * instantiated with the given combinations and is provided as extractor to a liblinear annotator.
     * The performance of each model is evaluated on the validation dataset. A detailed report on the performance of
     * each model is printed out.
     * @param patchSizes patch sizes
     * @param vocabularySizes vocabulary sizes
     * @param directionSteps direction steps
     * @throws IOException
     */
    public static void trainEvaluateModels(List<Integer> patchSizes, List<Integer> vocabularySizes, List<Integer> directionSteps) throws IOException {
        String path = "C:\\Users\\tamaj\\OneDrive\\University of Southampton\\University-Of-Southampton-CourseWork" +
                "\\Third_Year\\ComputerVision\\Coursework_1\\OpenIMAJ-Tutorials\\training\\DenselySampledPatchesBvW\\";

        GroupedDataset<String, VFSListDataset<FImage>, FImage> trainDataset = Dataset.getTrainDataset();
        int counter = 1;
        for(int vocSize : vocabularySizes){
            for(int patchSize : patchSizes){
                for(int directionStep : directionSteps) {

                    String bvwPath = path + "\\bvw" + counter;
                    DenselySampledPatchesBvW<String, ListDataset<FImage>> extractor = new DenselySampledPatchesBvW<>(patchSize, patchSize,
                            directionStep, directionStep, vocSize, 750, bvwPath, GroupedUniformRandomisedSampler.sample(trainDataset, 1500));
                    GroupedRandomSplitter<String, FImage> splits =
                            new GroupedRandomSplitter<>(trainDataset, 80, 0, 20);
                    trainEvaluateModel(extractor, SolverType.L2R_L2LOSS_SVC, 1, 0.00001, splits.getTrainingDataset(),
                            splits.getTestDataset());
                    counter++;
                }
            }
        }
    }

    /**
     * Train a set of linear classifiers on the given train dataset.
     * Features from the train dataset are extracted by the DenselySampledPatchesBvW extractor instance provided as
     * input. The performance of the trained model is evaluated on the given validation dataset. A detailed report on
     * the performance of the trained model is printed out.
     * @param extractor feature extractor
     * @param solverType liblinear solver
     * @param C C parameter
     * @param eps epsilon value
     * @param trainDataset train dataset
     * @param valDataset validation dataset
     * @return accuracy on the validation dataset
     */
    public static double trainEvaluateModel(DenselySampledPatchesBvW<String, ListDataset<FImage>> extractor,
                                            SolverType solverType, int C, double eps,
                                            GroupedDataset<String, ListDataset<FImage>, FImage> trainDataset,
                                            GroupedDataset<String, ListDataset<FImage>, FImage> valDataset) {

        // Create a set of linear classifiers
        LiblinearAnnotator<FImage, String> ann = new LiblinearAnnotator<>(
                extractor, LiblinearAnnotator.Mode.MULTICLASS, solverType, C, eps);
        // Train the set of linear classifiers
        ann.train(trainDataset);

        ClassificationEvaluator<CMResult<String>, String, FImage> eval = new ClassificationEvaluator<>(ann,
                valDataset, new CMAnalyser<>(CMAnalyser.Strategy.SINGLE));
        // Evaluate the performance of the trained model
        Map<FImage, ClassificationResult<String>> guesses = eval.evaluate();
        CMResult<String> result = eval.analyse(guesses);
        System.out.println(result.getDetailReport());
        // Return the accuracy achieved on the validation dataset
        return result.getMatrix().getAccuracy();
    }


    /**
     * Perform K-fold cross validation on a given dataset for a set of linear classifiers.
     * Features from the train dataset are extracted by the DenselySampledPatchesBvW extractor according to the
     * provided parameters.
     * @param w width of the window
     * @param h height of the window
     * @param stepX step along the x direction
     * @param stepY step along the y direction
     * @param visualWords number of visual words
     * @param samples number of samples drawn from the training dataset to be used for the creation of the codebook
     * @param nfolds number of folds
     * @param dataset dataset
     * @return average accuracy among the K-fold cross validation iterations
     */
    public static double crossValidation(int w, int h, int stepX, int stepY, int visualWords, int samples, int nfolds,
                                         GroupedDataset<String,ListDataset<FImage>,FImage> dataset){
        GroupedKFold<String,FImage> kfold = new GroupedKFold<>(nfolds);
        CrossValidationIterable<GroupedDataset<String,ListDataset<FImage>,FImage>> iterator =
                kfold.createIterable(dataset);
        double accum = 0;
        // Evaluate the model for each iteration of the K-fold cross validation process
        for (var data : iterator){
            accum += evaluateModel(w, h, stepX, stepY, visualWords, samples, data.getTrainingDataset(),
                    data.getValidationDataset());
        }
        // Return avg accuracy among the K-fold cross validation iterations
        return accum / iterator.numberIterations();
    }
}
