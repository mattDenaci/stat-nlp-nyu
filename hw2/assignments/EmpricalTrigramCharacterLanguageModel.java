package nlp.assignments;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

import nlp.langmodel.LanguageModel;
import nlp.util.Counter;
import nlp.util.CounterMap;

/**
 * A dummy language model -- uses empirical unigram counts, plus a single
 * ficticious count for unknown words.
 */
class EmpricalTrigramCharacterLanguageModel implements LanguageModel {

    static final String START = "<S>";
    static final String STOP = "</S>";
    static final String UNKNOWN = "*UNKNOWN*";
    static final double lambda1 = 0.5;
    static final double lambda2 = 0.3;

    Counter<String> wordCounter = new Counter<String>();
    CounterMap<String, String> bigramCounter = new CounterMap<String, String>();
    CounterMap<String, String> trigramCounter = new CounterMap<String, String>();

    public double getTrigramProbability(String prePreviousCharacter,
                                        String previousCharacter,
                                        String currentCharacter) {

        double trigramCount = trigramCounter.getCount(prePreviousCharacter
                + previousCharacter, currentCharacter);
        double bigramCount = bigramCounter.getCount(previousCharacter, currentCharacter);
        double unigramCount = wordCounter.getCount(currentCharacter);
        if (unigramCount == 0) {
            unigramCount = wordCounter.getCount(UNKNOWN);
        }
        return lambda1 * trigramCount + lambda2 * bigramCount
                + (1.0 - lambda1 - lambda2) * unigramCount;
    }

    public double getSentenceProbability(List<String> sentence) {
        List<String> stoppedSentence = new ArrayList<String>(sentence);

        double probability = 1.0;
        String prePreviousCharacter = START;
        String previousCharacter = START;
        String currentCharacter;

        for (int i = 0; i < stoppedSentence.size(); i++){
            String word = stoppedSentence.get(i);
            for(char character : word.toCharArray()) {
                currentCharacter = String.valueOf(character);
                probability *= getTrigramProbability(prePreviousCharacter, previousCharacter,
                        currentCharacter);
                prePreviousCharacter = previousCharacter;
                previousCharacter = currentCharacter;
            }
        }

        probability *= getTrigramProbability(prePreviousCharacter, previousCharacter, STOP);
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

    public EmpricalTrigramCharacterLanguageModel(
            Collection<List<String>> sentenceCollection) {
        for (List<String> sentence : sentenceCollection) {
            String previousLetter = START;
            String prePreviousLetter = START;
            List<String> stoppedSentence = new ArrayList<String>(sentence);
            for (int i = 0; i < stoppedSentence.size(); i++) {
                String word = stoppedSentence.get(i);
                   for(char charater : word.toCharArray()){
                       String currentLetter = String.valueOf(charater);
                       //System.out.println("Curr:"+ currentLetter);
                       //System.out.println("previous:"+ previousLetter);
                       //System.out.println("pre previous:"+ prePreviousLetter);
                       wordCounter.incrementCount(currentLetter, 1.0);
                       bigramCounter.incrementCount(previousLetter, currentLetter, 1.0);
                       trigramCounter.incrementCount(prePreviousLetter + previousLetter,
                               currentLetter, 1.0);
                       prePreviousLetter = previousLetter;
                       previousLetter = currentLetter;
                   }
            }
            bigramCounter.incrementCount(previousLetter,STOP, 1.0);
            trigramCounter.incrementCount(prePreviousLetter+previousLetter, STOP, 1.0);

        }
        wordCounter.incrementCount(UNKNOWN, 1.0);
        normalizeDistributions();
    }

    private void normalizeDistributions() {
        for (String previousBigram : trigramCounter.keySet()) {
            trigramCounter.getCounter(previousBigram).normalize();
        }
        for (String previousWord : bigramCounter.keySet()) {
            bigramCounter.getCounter(previousWord).normalize();
        }
        wordCounter.normalize();
    }
}
