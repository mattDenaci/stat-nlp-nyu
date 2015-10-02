package nlp.assignments;

import java.util.*;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

import nlp.classify.*;
import nlp.util.CommandLineUtils;
import nlp.util.Counter;
import nlp.util.CounterMap;
import org.apache.commons.math3.stat.correlation.PearsonsCorrelation;

/**
 * This is the main harness for assignment 2. To run this harness, use
 * <p/>
 * java nlp.assignments.ProperNameTester -path ASSIGNMENT_DATA_PATH -model
 * MODEL_DESCRIPTOR_STRING
 * <p/>
 * First verify that the data can be read on your system using the baseline
 * model. Second, find the point in the main method (near the bottom) where a
 * MostFrequentLabelClassifier is constructed. You will be writing new
 * implementations of the ProbabilisticClassifer interface and constructing them
 * there.
 */
public class ProperNameTester {

	public static class ProperNameFeatureExtractor implements
			FeatureExtractor<String, String> {

		/**
		 * This method takes the list of characters representing the proper name
		 * description, and produces a list of features which represent that
		 * description. The basic implementation is that only character-unigram
		 * features are extracted. An easy extension would be to also throw
		 * character bigrams into the feature list, but better features are also
		 * possible.
		 */
		public Counter<String> extractFeatures(String name) {
			char[] characters = name.toCharArray();

			Counter<String> features = new Counter<String>();
			//add character unigram features
			for (int i = 0; i < characters.length; i++) {
				char character = characters[i];
				//features.incrementCount("UNI-" + character, 1.0);
				if(i == 0)
					features.incrementCount("begins-with"+character,1);
				if(i == characters.length-1)
					features.incrementCount("ends-with"+character+characters[i], 1);
			}

			for(String word : name.split("\\s+")){
				if(word.length() > 1)
					features.incrementCount("word-"+word,1);
			}

			//extract bigrams
			for (int i = 0; i < characters.length-1; i++){
				if(i == 0 && characters.length >= 3)
					features.incrementCount("Bigram-begin"+ characters[i] + characters[i+1], 1);
				features.incrementCount("BIGRAM-" + characters[i] + characters[i+1], 1);
				if(i == characters.length-2)
					features.incrementCount("BIGRAM-end" + characters[i] + characters[i+1], 1);
			}
			//extract trigrams
			for (int i = 0; i < characters.length-2; i++) {
				if (i == 0 && characters.length >= 3)
					features.incrementCount("Tri-begin" + characters[i] + characters[i + 1] + characters[i + 2], 1);
				features.incrementCount("TRIG-" + characters[i] + characters[i + 1] + characters[i + 2], 1);
				if (i == characters.length - 3)
					features.incrementCount("TRIG-end" + characters[i] + characters[i + 1] + characters[i + 2], 1);
			}

			return features;
		}
	}

	private static List<LabeledInstance<String, String>> loadData(
			String fileName) throws IOException {
		BufferedReader reader = new BufferedReader(new FileReader(fileName));
		List<LabeledInstance<String, String>> labeledInstances = new ArrayList<LabeledInstance<String, String>>();
		while (reader.ready()) {
			String line = reader.readLine();
			String[] parts = line.split("\t");
			String label = parts[0];
			String name = parts[1];
			LabeledInstance<String, String> labeledInstance = new LabeledInstance<String, String>(
					label, name);
			labeledInstances.add(labeledInstance);
		}
		reader.close();
		return labeledInstances;
	}

	private static void testClassifier(
			ProbabilisticClassifier<String, String> classifier,
			List<LabeledInstance<String, String>> testData, boolean verbose) {
		double numCorrect = 0.0;
		double numTotal = 0.0;
		//in our confusion matrix we'll keep number of cor
		CounterMap<String, String> confusionMatrix = new CounterMap<String, String>();
		double[] accuracyArray = new double[testData.size()];
		double[] confidenceArray = new double[testData.size()];
		int datumIndex = 0;
		for (LabeledInstance<String, String> testDatum : testData) {

			String name = testDatum.getInput();
			String label = classifier.getLabel(name);
			
			double confidence = classifier.getProbabilities(name).getCount(
					label);
			confidenceArray[datumIndex] = confidence;

			if (label.equals(testDatum.getLabel())) {
				numCorrect += 1.0;
				confusionMatrix.incrementCount(label, label, 1.0);
				accuracyArray[datumIndex] = 1;
			} else {
				confusionMatrix.incrementCount(testDatum.getLabel(), label, 1.0);
				accuracyArray[datumIndex] = 0;

				if (verbose) {
					// display an error
					System.err.println("Example:\t" + name + " guess=" + label
							+ " gold=" + testDatum.getLabel() + " confidence="
							+ confidence);
				}
			}
			numTotal += 1.0;
			datumIndex += 1.0;
		}
		double accuracy = numCorrect / numTotal;
		System.out.println("Accuracy: " + accuracy);
		printConfusionMatrix(confusionMatrix);
		PearsonsCorrelation pc = new PearsonsCorrelation();
		//System.out.println("Accuracy Array"+ Arrays.toString(accuracyArray));
		//System.out.println("Confidence Array"+ Arrays.toString(confidenceArray));
		System.out.println("Correlation of Accuracy and Confidence: " + pc.correlation(accuracyArray, confidenceArray));
	}

	private static void printConfusionMatrix(CounterMap<String, String> confusionMatrix) {
		System.out.println("Confusion Matrix: ");
		System.out.println("counts");
		System.out.println(confusionMatrix.toString());
		System.out.println("nomalized");
		confusionMatrix.normalize();
		System.out.println(confusionMatrix.toString());
	}


	public static void main(String[] args) throws IOException {
		// Parse command line flags and arguments
		Map<String, String> argMap = CommandLineUtils
				.simpleCommandLineParser(args);

		// Set up default parameters and settings
		String basePath = ".";
		String model = "baseline";
		boolean verbose = false;
		boolean useValidation = true;

		// Update defaults using command line specifications

		// The path to the assignment data
		if (argMap.containsKey("-path")) {
			basePath = argMap.get("-path");
		}
		System.out.println("Using base path: " + basePath);

		// A string descriptor of the model to use
		if (argMap.containsKey("-model")) {
			model = argMap.get("-model");
		}
		System.out.println("Using model: " + model);

		// A string descriptor of the model to use
		if (argMap.containsKey("-test")) {
			String testString = argMap.get("-test");
			if (testString.equalsIgnoreCase("test"))
				useValidation = false;
		}
		System.out.println("Testing on: "
				+ (useValidation ? "validation" : "test"));

		// Whether or not to print the individual speech errors.
		if (argMap.containsKey("-verbose")) {
			verbose = true;
		}

		// Load training, validation, and test data
		List<LabeledInstance<String, String>> trainingData = loadData(basePath
				+ "/pnp-train.txt");
		List<LabeledInstance<String, String>> validationData = loadData(basePath
				+ "/pnp-validate.txt");
		List<LabeledInstance<String, String>> testData = loadData(basePath
				+ "/pnp-test.txt");

		// Learn a classifier
		ProbabilisticClassifier<String, String> classifier = null;
		if(model.equalsIgnoreCase("all")){
			System.out.println("Base line: Most Frequent Label Classifier");
			classifier = new MostFrequentLabelClassifier.Factory<String, String>()
					.trainClassifier(trainingData);
			testClassifier(classifier, (useValidation ? validationData : testData),
					verbose);
			System.out.println("Unigram Character Label Classifier");
			ProbabilisticClassifierFactory<String, String> factory = new WordUnigramClassifier.Factory<String, String>();
			classifier = factory.trainClassifier(trainingData);
			testClassifier(classifier, (useValidation ? validationData : testData),
					verbose);
			//((WordUnigramClassifier)classifier).listWordsForCatagory();
			System.out.println("Trigram Character Label Classifier");
			// this is just a trigram
			factory = new TrigramCharacterLanguageClassifier.Factory<String, String>();
			classifier = factory.trainClassifier(trainingData);
			//((TrigramCharacterLanguageClassifier)classifier).characterListPerCatagory();
			testClassifier(classifier, (useValidation ? validationData : testData),
					verbose);
			System.out.println("Max ent classifier");
			factory = new MaximumEntropyClassifier.Factory<String, String, String>(
					1.5, 50, new ProperNameFeatureExtractor());
			classifier = factory.trainClassifier(trainingData);
			testClassifier(classifier, (useValidation ? validationData : testData),
					verbose);
			System.out.println("Perceptron Single Iteration");
			factory = new PerceptronClassifier.Factory<String, String,String>(
					1.0, 1, new ProperNameFeatureExtractor());
			classifier = factory.trainClassifier(trainingData);
			testClassifier(classifier, (useValidation ? validationData : testData),
					verbose);
			System.out.println("Perceptron 10 Iteration");
			factory = new PerceptronClassifier.Factory<String, String,String>(
					10.0, 1, new ProperNameFeatureExtractor());
			classifier = factory.trainClassifier(trainingData);
			testClassifier(classifier, (useValidation ? validationData : testData),
					verbose);

			System.out.println("Done!");
			return;
		}



		if (model.equalsIgnoreCase("baseline")) {
			classifier = new MostFrequentLabelClassifier.Factory<String, String>()
					.trainClassifier(trainingData);
		} else if (model.equalsIgnoreCase("n-gram")) {
			// this is just a trigram
			ProbabilisticClassifierFactory<String, String> factory = new TrigramCharacterLanguageClassifier.Factory<String, String>();
			classifier = factory.trainClassifier(trainingData);
		} else if (model.equalsIgnoreCase("maxent")) {
			ProbabilisticClassifierFactory<String, String> factory = new MaximumEntropyClassifier.Factory<String, String, String>(
					1.5, 40, new ProperNameFeatureExtractor());
			classifier = factory.trainClassifier(trainingData);
		} else if(model.equalsIgnoreCase("word-uni-gram")){
			ProbabilisticClassifierFactory<String, String> factory = new WordUnigramClassifier.Factory<String, String>();
			classifier = factory.trainClassifier(trainingData);
			((WordUnigramClassifier)classifier).listWordsForCatagory();
		} else if (model.equalsIgnoreCase("perceptron")){
			ProbabilisticClassifierFactory<String, String> factory = new PerceptronClassifier.Factory<String, String,String>(
				1.0, 1, new ProperNameFeatureExtractor());
			classifier = factory.trainClassifier(trainingData);

		} else {
			throw new RuntimeException("Unknown model descriptor: " + model);
		}

		// Test classifier
		testClassifier(classifier, (useValidation ? validationData : testData),
				verbose);
	}
}
