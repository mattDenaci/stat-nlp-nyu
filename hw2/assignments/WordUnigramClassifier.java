package nlp.assignments;

import nlp.classify.*;
import nlp.util.Counter;

import java.util.*;

public class WordUnigramClassifier<I,L> implements
        ProbabilisticClassifier<I,L> {

    private  Map<String, EmpiricalUnigramLanguageModel> languageModels;

    public static class Factory<I, L> implements ProbabilisticClassifierFactory<I, L> {

        public ProbabilisticClassifier<I, L> trainClassifier(
                List<LabeledInstance<I, L>> trainingData) {
            //TODO: if performance is bad replace with array soluation
            Map<L, Collection<List<String>>> labelToInstaceList = new HashMap<L, Collection<List<String>>>();

            for (LabeledInstance<I, L> labledInstance : trainingData) {
                //TODO: think about how to make Emprical Unigram Language model abstract
                Collection<List<String>> sentenceListForLabel;
                if (labelToInstaceList.containsKey(labledInstance.getLabel())) {
                    sentenceListForLabel = labelToInstaceList.get(labledInstance.getLabel());
                } else {
                    sentenceListForLabel = new ArrayList<List<String>>();
                }
                String input = (String) labledInstance.getInput();
                List<String> sentence = new ArrayList<String>();
                for (String word : input.split("\\s+"))
                    sentence.add(word.toLowerCase());
                sentenceListForLabel.add(sentence);
                labelToInstaceList.put(labledInstance.getLabel(), sentenceListForLabel);

            }

            Map<String, EmpiricalUnigramLanguageModel> lms = new HashMap<String, EmpiricalUnigramLanguageModel>();

            for (Map.Entry<L, Collection<List<String>>> entry : labelToInstaceList.entrySet()) {
                lms.put((String) entry.getKey(), new EmpiricalUnigramLanguageModel(entry.getValue()));
            }

            return new WordUnigramClassifier<I, L>(lms);
        }
    }

    public Counter<L> getProbabilities(I input) {
        List<String> place = new ArrayList<String>();
        for(String word : ((String)input).split("\\s+")){
            place.add(word);
        }

        Counter<L> probabiltyCounter = new Counter<L>();
        for(Map.Entry<String, EmpiricalUnigramLanguageModel> entry : languageModels.entrySet()){
            //TODO: fix sloppy generic stuff
            L label = (L)entry.getKey();
            double probability = entry.getValue().getSentenceProbability(place);
            probabiltyCounter.setCount(label, probability);
        }
        return probabiltyCounter;
    }


    public void listWordsForCatagory() {
        for (Map.Entry<String, EmpiricalUnigramLanguageModel> entry : languageModels.entrySet()) {
            System.out.println("Word counts for: " + entry.getKey());
            System.out.println(entry.getValue().wordCounter);
        }
    }

    public L getLabel(I input) {
        return getProbabilities(input).argMax();
    }

    public WordUnigramClassifier(Map<String, EmpiricalUnigramLanguageModel> languageModels) {
        this.languageModels = languageModels;
    }

}
