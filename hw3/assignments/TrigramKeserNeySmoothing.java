package nlp.assignments;

import nlp.langmodel.LanguageModel;
import nlp.util.Counter;
import nlp.util.CounterMap;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

/**
 * Created by mattdenaci on 9/16/15.
 */
public class TrigramKeserNeySmoothing implements LanguageModel {
    static final String START = "<S>";
    static final String STOP = "</S>";
    static final String UNKNOWN = "*UNKNOWN*";


    Counter<String> wordCounter = new Counter<String>();
    Counter<String> continuationCounter = new Counter<String>();
    CounterMap<String, String> bigramCounter = new CounterMap<String, String>();
    CounterMap<String, String> trigramCounter = new CounterMap<String, String>();


    public double lambda1 = 0.5;
    public double lambda2 = 0.3;



    public double getTrigramProb(String prePreviousWord, String previousWord, String word) {
        double bigramProbability = bigramCounter.getCount(previousWord, word);
        double trigramProbability = trigramCounter.getCount(prePreviousWord + previousWord, word);
        double continuationProbablitity = continuationCounter.getCount(word);
        if (continuationProbablitity == 0) {
            continuationProbablitity = continuationCounter.getCount(UNKNOWN);
        }

        return lambda1 * trigramProbability + lambda2 * bigramProbability + (1 - lambda1 - lambda2) * continuationProbablitity;
    }

    public double getSentenceProbability(List<String> sentence) {
        List<String> stoppedSentence = new ArrayList<String>(sentence);
        stoppedSentence.add(0, START);
        stoppedSentence.add(0, START);
        stoppedSentence.add(STOP);
        double probability = 1.0;
        String prePreviousWord = stoppedSentence.get(0);
        String previousWord = stoppedSentence.get(1);
        for (int i = 2; i < stoppedSentence.size(); i++) {
            String word = stoppedSentence.get(i);
            probability *= getTrigramProb(prePreviousWord, previousWord,
                    word);
            prePreviousWord = previousWord;
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

    public TrigramKeserNeySmoothing(
            Collection<List<String>> sentenceCollection, double lambda1, double lambda2) {
        this.lambda1 = lambda1;
        this.lambda2 = lambda2;
        for (List<String> sentence : sentenceCollection) {
            List<String> stoppedSentence = new ArrayList<String>(sentence);
            stoppedSentence.add(0, START);
            stoppedSentence.add(0, START);
            stoppedSentence.add(STOP);
            String prePreviousWord = stoppedSentence.get(0);
            String previousWord = stoppedSentence.get(1);
            for (int i = 2; i < stoppedSentence.size(); i++) {
                String word = stoppedSentence.get(i);
                wordCounter.incrementCount(word, 1.0);
                trigramCounter.incrementCount(prePreviousWord + previousWord, word, 1.0);
                bigramCounter.incrementCount(previousWord, word, 1.0);
                prePreviousWord = previousWord;
                previousWord = word;
            }
        }

        for(String word : bigramCounter.keySet()){
            Counter<String> bigramCountsForWord = trigramCounter.getCounter(word);
            continuationCounter.setCount(word, bigramCountsForWord.size());
        }

        continuationCounter.incrementCount(UNKNOWN, 1.0);
        normalizeDistributions();
    }


    private void normalizeDistributions() {
        for (String previousWord : bigramCounter.keySet()) {
            bigramCounter.getCounter(previousWord).normalize();
        }
        for (String previousWords : trigramCounter.keySet()){
            trigramCounter.getCounter(previousWords).normalize();
        }
        continuationCounter.normalize();
        wordCounter.normalize();
    }
}


