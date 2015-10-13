package nlp.assignments;

import nlp.classify.FeatureExtractor;
import nlp.util.Counter;

/**
 * Created by mattdenaci on 9/30/15.
 * A simple word feature extractor that splits on whitespace
 */
public class WordFeatureExtractor implements FeatureExtractor<String,String> {
    @Override
    public Counter<String> extractFeatures(String instance) {
        Counter<String> features = new Counter<String>();
        for(String word : instance.split("\\s+")){
            if(word.length() > 1)
                features.incrementCount("word-"+word,1);
        }
        return features;
    }

}
