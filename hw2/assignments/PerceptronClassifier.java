package nlp.assignments;

import nlp.classify.*;
import nlp.math.*;
import nlp.util.Counter;
import nlp.util.Indexer;
import nlp.util.Pair;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

/**
 * Created by mattdenaci on 9/23/15.
 */
public class PerceptronClassifier<I, F, L> implements
        ProbabilisticClassifier<I, L> {

    /**
     * Factory for training MaximumEntropyClassifiers.
     */
    public static class Factory<I, F, L> implements
            ProbabilisticClassifierFactory<I, L> {

        double sigma;
        int iterations;
        FeatureExtractor<I, F> featureExtractor;

        public ProbabilisticClassifier<I, L> trainClassifier(
                List<LabeledInstance<I, L>> trainingData) {


            Encoding<F, L> encoding = buildEncoding(trainingData);
            IndexLinearizer indexLinearizer = buildIndexLinearizer(encoding);
            double[] weights = buildInitialWeights(indexLinearizer);

            //perceptron training step
            EncodedDatum[] data = encodeData(trainingData, encoding);
            double[] averageWeights = DoubleArrays.constantArray(0.0, weights.length);
            for(int i = 0; i < iterations; i++)
                for(EncodedDatum datum : doTheFisherYatesShuffle(data)) {
                   double[] curWeights = updateWeights(datum, weights, encoding, indexLinearizer);
                   DoubleArrays.addDestructive(averageWeights, curWeights);
                }

            DoubleArrays.scale(averageWeights, 1.0/(data.length*iterations));

            return new PerceptronClassifier<I, F, L>(weights, encoding,
                    indexLinearizer, featureExtractor);
        }
        //TODO: templatize and put into it's own class similar to double arrays Maybe 'ObjectArrays' or TypeArrays or something
        public static EncodedDatum [] doTheFisherYatesShuffle(EncodedDatum[] x)
        {
            EncodedDatum[] y = new EncodedDatum[x.length];
            System.arraycopy(x, 0, y, 0, x.length);

            Random rnd = new Random();
            for (int i = y.length - 1; i > 0; i--)
            {
                int index = rnd.nextInt(i + 1);
                EncodedDatum a = y[index];
                y[index] = y[i];
                y[i] = a;
            }
            return y;
        }

        private double[] updateWeights(EncodedDatum datum, double[] weights, Encoding encoding,
                                   IndexLinearizer indexLinearizer) {
            double[] probablities = DoubleArrays.exponentiate(
                    getLogProbabilities(datum, weights, encoding, indexLinearizer));
            int predictedLabelIndex = DoubleArrays.argMax(probablities);

            if(datum.getLabelIndex() != predictedLabelIndex){
                for(int activeFeature = 0; activeFeature < datum.getNumActiveFeatures(); activeFeature++){
                    weights[indexLinearizer.getLinearIndex(
                            datum.getFeatureIndex(activeFeature), datum.getLabelIndex())
                            ] += datum.getFeatureCount(activeFeature);
                    weights[indexLinearizer.getLinearIndex(
                            datum.getFeatureIndex(activeFeature), predictedLabelIndex)
                            ] -= datum.getFeatureCount(activeFeature);
                }
            }
            return weights;
        }

        private double[] buildInitialWeights(IndexLinearizer indexLinearizer) {
            return DoubleArrays.constantArray(0.0,
                    indexLinearizer.getNumLinearIndexes());
        }

        private IndexLinearizer buildIndexLinearizer(Encoding<F, L> encoding) {
            return new IndexLinearizer(encoding.getNumFeatures(),
                    encoding.getNumLabels());
        }

        private Encoding<F, L> buildEncoding(List<LabeledInstance<I, L>> data) {
            Indexer<F> featureIndexer = new Indexer<F>();
            Indexer<L> labelIndexer = new Indexer<L>();
            for (LabeledInstance<I, L> labeledInstance : data) {
                L label = labeledInstance.getLabel();
                Counter<F> features = featureExtractor
                        .extractFeatures(labeledInstance.getInput());
                LabeledFeatureVector<F, L> labeledDatum = new BasicLabeledFeatureVector<F, L>(
                        label, features);
                labelIndexer.add(labeledDatum.getLabel());
                for (F feature : labeledDatum.getFeatures().keySet()) {
                    featureIndexer.add(feature);
                }
            }
            return new Encoding<F, L>(featureIndexer, labelIndexer);
        }

        private EncodedDatum[] encodeData(List<LabeledInstance<I, L>> data,
                                          Encoding<F, L> encoding) {
            EncodedDatum[] encodedData = new EncodedDatum[data.size()];
            for (int i = 0; i < data.size(); i++) {
                LabeledInstance<I, L> labeledInstance = data.get(i);
                L label = labeledInstance.getLabel();
                Counter<F> features = featureExtractor
                        .extractFeatures(labeledInstance.getInput());
                LabeledFeatureVector<F, L> labeledFeatureVector = new BasicLabeledFeatureVector<F, L>(
                        label, features);
                encodedData[i] = EncodedDatum.encodeLabeledDatum(
                        labeledFeatureVector, encoding);
            }
            return encodedData;
        }

        /**
         * Sigma controls the variance on the prior / penalty term. 1.0 is a
         * reasonable value for large problems, bigger sigma means LESS
         * smoothing. Zero sigma is a special indicator that no smoothing is to
         * be done.
         * <p/>
         * Iterations determines the maximum number of iterations the
         * optimization code can take before stopping.
         */
        public Factory(double sigma, int iterations,
                       FeatureExtractor<I, F> featureExtractor) {
            this.sigma = sigma;
            this.iterations = iterations;
            this.featureExtractor = featureExtractor;
        }
    }

    /**
     * EncodedDatums are sparse representations of (labeled) feature count
     * vectors for a given data point. Use getNumActiveFeatures() to see how
     * many features have non-zero count in a datum. Then, use getFeatureIndex()
     * and getFeatureCount() to retreive the number and count of each non-zero
     * feature. Use getLabelIndex() to get the label's number.
     */
    public static class EncodedDatum {

        public static <F, L> EncodedDatum encodeDatum(
                FeatureVector<F> featureVector, Encoding<F, L> encoding) {
            Counter<F> features = featureVector.getFeatures();
            Counter<F> knownFeatures = new Counter<F>();
            for (F feature : features.keySet()) {
                if (encoding.getFeatureIndex(feature) < 0)
                    continue;
                knownFeatures.incrementCount(feature,
                        features.getCount(feature));
            }
            int numActiveFeatures = knownFeatures.keySet().size();
            int[] featureIndexes = new int[numActiveFeatures];
            double[] featureCounts = new double[knownFeatures.keySet().size()];
            int i = 0;
            for (F feature : knownFeatures.keySet()) {
                int index = encoding.getFeatureIndex(feature);
                double count = knownFeatures.getCount(feature);
                featureIndexes[i] = index;
                featureCounts[i] = count;
                i++;
            }
            EncodedDatum encodedDatum = new EncodedDatum(-1, featureIndexes,
                    featureCounts);
            return encodedDatum;
        }

        public static <F, L> EncodedDatum encodeLabeledDatum(
                LabeledFeatureVector<F, L> labeledDatum, Encoding<F, L> encoding) {
            EncodedDatum encodedDatum = encodeDatum(labeledDatum, encoding);
            encodedDatum.labelIndex = encoding.getLabelIndex(labeledDatum
                    .getLabel());
            return encodedDatum;
        }

        int labelIndex;
        int[] featureIndexes;
        double[] featureCounts;

        public int getLabelIndex() {
            return labelIndex;
        }

        public int getNumActiveFeatures() {
            return featureCounts.length;
        }

        public int getFeatureIndex(int num) {
            return featureIndexes[num];
        }

        public double getFeatureCount(int num) {
            return featureCounts[num];
        }

        public EncodedDatum(int labelIndex, int[] featureIndexes,
                            double[] featureCounts) {
            this.labelIndex = labelIndex;
            this.featureIndexes = featureIndexes;
            this.featureCounts = featureCounts;
        }
    }

    /**
     * The Encoding maintains correspondences between the various representions
     * of the data, labels, and features. The external representations of labels
     * and features are object-based. The functions getLabelIndex() and
     * getFeatureIndex() can be used to translate those objects to integer
     * representatiosn: numbers between 0 and getNumLabels() or getNumFeatures()
     * (exclusive). The inverses of this map are the getLabel() and getFeature()
     * functions.
     */
    public static class Encoding<F, L> {
        Indexer<F> featureIndexer;
        Indexer<L> labelIndexer;

        public int getNumFeatures() {
            return featureIndexer.size();
        }

        public int getFeatureIndex(F feature) {
            return featureIndexer.indexOf(feature);
        }

        public F getFeature(int featureIndex) {
            return featureIndexer.get(featureIndex);
        }

        public int getNumLabels() {
            return labelIndexer.size();
        }

        public int getLabelIndex(L label) {
            return labelIndexer.indexOf(label);
        }

        public L getLabel(int labelIndex) {
            return labelIndexer.get(labelIndex);
        }

        public Encoding(Indexer<F> featureIndexer, Indexer<L> labelIndexer) {
            this.featureIndexer = featureIndexer;
            this.labelIndexer = labelIndexer;
        }
    }

    /**
     * The IndexLinearizer maintains the linearization of the two-dimensional
     * features-by-labels pair space. This is because, while we might think
     * about lambdas and derivatives as being indexed by a feature-label pair,
     * the optimization code expects one long vector for lambdas and
     * derivatives. To go from a pair featureIndex, labelIndex to a single
     * pairIndex, use getLinearIndex().
     */
    public static class IndexLinearizer {
        int numFeatures;
        int numLabels;

        public int getNumLinearIndexes() {
            return numFeatures * numLabels;
        }

        public int getLinearIndex(int featureIndex, int labelIndex) {
            return labelIndex + featureIndex * numLabels;
        }

        public int getFeatureIndex(int linearIndex) {
            return linearIndex / numLabels;
        }

        public int getLabelIndex(int linearIndex) {
            return linearIndex % numLabels;
        }

        public IndexLinearizer(int numFeatures, int numLabels) {
            this.numFeatures = numFeatures;
            this.numLabels = numLabels;
        }
    }

    private double[] weights;
    private Encoding<F, L> encoding;
    private IndexLinearizer indexLinearizer;
    private FeatureExtractor<I, F> featureExtractor;

    /**
     * Calculate the log probabilities of each class, for the given datum
     * (feature bundle). Note that the weighted votes (refered to as
     * activations) are *almost* log probabilities, but need to be normalized.
     */
    private static <F, L> double[] getLogProbabilities(EncodedDatum datum,
                                                       double[] weights, Encoding<F, L> encoding,
                                                       IndexLinearizer indexLinearizer) {

        double[] logProbabilities = new double[encoding.getNumLabels()];
        for(int labelIndex = 0; labelIndex < encoding.getNumLabels(); labelIndex++){
            for(int i = 0; i < datum.getNumActiveFeatures(); i++) {
                double featureCount = datum.getFeatureCount(i);
                int featureIndex = datum.getFeatureIndex(i);
                int weightFeatureIndex = indexLinearizer.getLinearIndex(featureIndex, labelIndex);
                //System.out.println("feature is: " + encoding.getFeature(featureIndex));
                //System.out.println("labelIndex:" + encoding.getLabel(labelIndex));
                //System.out.println("weight is "+ weights[weightFeatureIndex]*featureCount);
                logProbabilities[labelIndex] += weights[weightFeatureIndex]*featureCount;

            }
        }
        double normalizer = 0;
        for(int i = 0; i < logProbabilities.length; i++)
            normalizer += Math.exp(logProbabilities[i]);

        normalizer = Math.log(normalizer);

        for(int i = 0; i < logProbabilities.length; i++)
            logProbabilities[i] = logProbabilities[i] - normalizer;
        return logProbabilities;

    }

    public Counter<L> getProbabilities(I input) {
        FeatureVector<F> featureVector = new BasicFeatureVector<F>(
                featureExtractor.extractFeatures(input));
        return getProbabilities(featureVector);
    }

    private Counter<L> getProbabilities(FeatureVector<F> featureVector) {
        EncodedDatum encodedDatum = EncodedDatum.encodeDatum(featureVector,
                encoding);
        double[] logProbabilities = getLogProbabilities(encodedDatum, weights,
                encoding, indexLinearizer);
        return logProbabiltyArrayToProbabiltyCounter(logProbabilities);
    }

    private Counter<L> logProbabiltyArrayToProbabiltyCounter(
            double[] logProbabilities) {
        Counter<L> probabiltyCounter = new Counter<L>();
        for (int labelIndex = 0; labelIndex < logProbabilities.length; labelIndex++) {
            double logProbability = logProbabilities[labelIndex];
            double probability = Math.exp(logProbability);
            L label = encoding.getLabel(labelIndex);
            probabiltyCounter.setCount(label, probability);
        }
        return probabiltyCounter;
    }

    public L getLabel(I input) {
        return getProbabilities(input).argMax();
    }

    public PerceptronClassifier(double[] weights, Encoding<F, L> encoding,
                                IndexLinearizer indexLinearizer,
                                FeatureExtractor<I, F> featureExtractor) {
        this.weights = weights;
        this.encoding = encoding;
        this.indexLinearizer = indexLinearizer;
        this.featureExtractor = featureExtractor;
    }
}
