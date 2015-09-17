package nlp.assignments;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

import nlp.langmodel.LanguageModel;
import nlp.util.Counter;
import nlp.util.CounterMap;


class BigramKeserNeySmoothing implements LanguageModel {

	static final String START = "<S>";
	static final String STOP = "</S>";
	static final String UNKNOWN = "*UNKNOWN*";


	Counter<String> wordCounter = new Counter<String>();
	Counter<String> continuationCounter = new Counter<String>();
	Counter<String> followCounter = new Counter<String>();
	CounterMap<String, String> bigramCounter = new CounterMap<String, String>();
	CounterMap<String, String> revBigramCounter = new CounterMap<String, String>();


	double discount = 0.75;


	public double getBigramProbability(
			String previousWord, String word) {
		double bigramProbability = bigramCounter.getCount(previousWord, word);
		double continuationProbablitity = continuationCounter.getCount(word);
		if (continuationProbablitity == 0){
			continuationProbablitity = continuationCounter.getCount(UNKNOWN);
		}

		return bigramProbability + lambda(previousWord)*continuationProbablitity;

	}

	private double lambda(String previousWord) {
		double previousWordWordCount = wordCounter.getCount(previousWord);
		previousWordWordCount = previousWordWordCount > 0 ? previousWordWordCount : wordCounter.getCount(UNKNOWN);
		double followCount  = followCounter.getCount(previousWord);
		followCount = followCount > 0 ? followCount : followCounter.getCount(UNKNOWN);
		return (discount/previousWordWordCount) * followCount;
	}

	public double getSentenceProbability(List<String> sentence) {
		List<String> stoppedSentence = new ArrayList<String>(sentence);
		stoppedSentence.add(0, START);
		stoppedSentence.add(STOP);
		double probability = 1.0;
		String previousWord = stoppedSentence.get(0);
		for (int i = 1; i < stoppedSentence.size(); i++) {
			String word = stoppedSentence.get(i);
			probability *= getBigramProbability(previousWord, word);
			previousWord = word;
		}
		return probability;
	}



	String generateWord() {
		double sample = Math.random();
		double sum = 0.0;
		for (String word : wordCounter.keySet()) {
			sum += wordCounter.getCount(word);
			if (sum > sample) {
				return word;
			}
		}
		return UNKNOWN;
	}

	public List<String> generateSentence() {
		List<String> sentence = new ArrayList<String>();
		String word = generateWord();
		while (!word.equals(STOP)) {
			sentence.add(word);
			word = generateWord();
		}
		return sentence;
	}

	public BigramKeserNeySmoothing(
			Collection<List<String>> sentenceCollection, double discount) {
		this.discount = discount;
		for (List<String> sentence : sentenceCollection) {
			List<String> stoppedSentence = new ArrayList<String>(sentence);
			stoppedSentence.add(0, START);
			stoppedSentence.add(STOP);
			String previousWord = stoppedSentence.get(0);
			for (int i = 1; i < stoppedSentence.size(); i++) {
				String word = stoppedSentence.get(i);
				wordCounter.incrementCount(word, 1.0);
				bigramCounter.incrementCount(previousWord, word, 1.0);
				revBigramCounter.incrementCount(word, previousWord, 1.0);
				previousWord = word;
			}
		}

		for(String word : bigramCounter.keySet()){
			Counter<String> bigramCountsForWord = bigramCounter.getCounter(word);
			continuationCounter.setCount(word, bigramCountsForWord.size());
		}

		for(String word: revBigramCounter.keySet()){
			Counter<String> revBigramCountsForWord = revBigramCounter.getCounter(word);
			followCounter.setCount(word, revBigramCountsForWord.size());
		}
		continuationCounter.incrementCount(UNKNOWN, 1.0);
		wordCounter.incrementCount(UNKNOWN, 1.0);
		followCounter.incrementCount(UNKNOWN, 1.0);
		//TODO: think about what do do about follow counter. Should we (i) normalize it
		//TODO: and (ii) should we adj for unknown worlds
		bigramCounter.discount(discount);
		normalizeDistributions();
	}


	private void normalizeDistributions() {
		for (String previousWord : bigramCounter.keySet()) {
			bigramCounter.getCounter(previousWord).normalize();
		}
		//wordCounter.normalize();
		//followCounter.normalize();
		continuationCounter.normalize();
	}
}